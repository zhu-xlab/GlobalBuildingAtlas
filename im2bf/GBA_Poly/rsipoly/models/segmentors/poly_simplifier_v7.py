# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import numpy as np
import mmcv
from tqdm import tqdm
import cv2
import random
from scipy.spatial import cKDTree

from rsipoly.core import add_prefix
from rsipoly.ops import resize
from .. import builder
from ..builder import SEGMENTORS
# from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from mmcv.parallel import DataContainer
from pycocotools import mask as cocomask
from rsidet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmcv.runner import BaseModule
import torch.distributed as dist
from collections import OrderedDict
from rasterio.features import shapes
import shapely
import geojson
import rasterio
import networkx

class ScoreNet(nn.Module):

    def __init__(self, in_ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        n_points = x.shape[-1]

        x = x.unsqueeze(-1)
        x = x.repeat(1,1,1,n_points)
        t = torch.transpose(x, 2, 3)
        x = torch.cat((x, t), dim=1)

        x = self.conv1(x)
        # x = self.conv1(masked_x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        return x[:,0]

@SEGMENTORS.register_module()
class PolySimplifierV7(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self, backbone, decode_head, simp_net, assigner, sampler, score_net=None, wandb_cfg=None,
        window_feat_size=6, len_sampled_segments=50, noise_conf=None, linear_conf=None,
        ring_reg_net=None, ring_cls_net=None, ring_angle_net=None, dice_loss_conf=None, ring_sample_conf=None,
        loss_weights=None, **kwargs,
    ):
        super(PolySimplifierV7, self).__init__(backbone, decode_head, **kwargs)
        self.backbone = builder.build_backbone(backbone)
        self.simp_net = builder.build_backbone(simp_net)
        self.score_net = ScoreNet(**score_net) if score_net is not None else None

        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler)
        self.wandb_cfg = wandb_cfg
        self.nms_width = 7
        self.cur_iter = 0
        self.window_feat_size = window_feat_size
        self.len_sampled_segments = len_sampled_segments
        self.l1_loss_fun = nn.SmoothL1Loss(reduction='none')
        self.cse_loss_fun = nn.CrossEntropyLoss(reduction='none')
        self.linear = nn.Linear(linear_conf['in_channels'], linear_conf['out_channels'])
        self.ring_reg_net = nn.Linear(ring_reg_net['in_channels'], ring_reg_net['out_channels'])
        self.ring_cls_net = nn.Linear(ring_cls_net['in_channels'], ring_cls_net['out_channels'])
        self.ring_angle_net = nn.Linear(ring_angle_net['in_channels'], ring_angle_net['out_channels'])
        self.loss_weights = loss_weights

        self.dice_loss_fun = None
        if dice_loss_conf:
            self.dice_loss_fun = builder.build_loss(dice_loss_conf)
        self.noise_conf = noise_conf
        self.ring_sample_conf = ring_sample_conf

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        state = None
        if type(out) == tuple:
            out, state = out
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return out if state is None else (out, state)

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, state = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses, state

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        states = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux, state_aux = aux_head.forward_train(x, img_metas,
                                                             gt_semantic_seg,
                                                             self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
                states.update(add_prefix(state_aux, f'aux_{idx}'))
        else:
            loss_aux, state_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
            states.update(add_prefix(state_aux, 'aux'))

        return losses, states

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # polygon = data_batch['polygon']
        # B, C, H, W = data_batch['img'].shape
        self.cur_iter += 1

        losses, states = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        # img = data_batch['img']
        # gt_semantic_seg = data_batch['gt_semantic_seg']
        # vis_mask = F.interpolate(gt_semantic_seg, size=(H, W))

        if 'img' in data_batch:
            states['vis|masks_gt'] = [data_batch['img'], data_batch['gt_semantic_seg'][:, 0].cpu().numpy()]

        if self.noise_conf:
            states['vis|masks_noise_img_gt'] = [states['noise_imgs'], data_batch['gt_semantic_seg'][:, 0].cpu().numpy()]

        if 'seg_logits' in states:
            probs = F.softmax(states['seg_logits'], dim=1)[:, 1:, ...].detach()
            states['vis|featmap_probs'] = [data_batch['img'], probs]
            # states['vis|points_vertices'] = [data_batch['img'], data_batch['contours']]
            # states['vis|points_pos_point_preds'] = [data_batch['img'], states['pos_point_preds']]

        """
        if 'point_preds_ori' in states:
            point_preds = states['point_preds_ori'].detach()
            states['vis|points_point_preds'] = [data_batch['img'], point_preds.view(1, -1, 2)]
        """

        if 'gdal_polygons' in states:
            polygons = states['gdal_polygons']
            states['vis|polygons_gdal_polygons'] = [data_batch['img'], polygons]

        if 'gt_polygons' in states:
            polygons = states['gt_polygons']
            states['vis|polygons_gt_polygons'] = [data_batch['img'], polygons]

        if 'raster_gt' in states:
            raster_gt = states['raster_gt']
            states['vis|super_pixel'] = [raster_gt]

        if 'comp_points' in states:
            states['vis|points_super_pixel'] = [data_batch['img'], states['comp_points'], states['comp_labels']]

        if 'super_pixel_masks' in states:
            states['vis|super_pixel_preds'] = [states['super_pixel_masks']]

        if 'pred_rings' in states:
            states['vis|polygons_pred_rings'] = [None, states['pred_rings']]

        if 'gt_rings' in states:
            states['vis|polygons_gt_rings'] = [None, states['gt_rings']]

        if 'next_rings' in states:
            states['vis|polygons_next_rings'] = [None, states['next_rings']]


        """
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states
        )
        """

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states
        )

        return outputs

    def get_window_feats(self, img, points, w):
        _, H, W = img.shape
        N, _ = points.shape
        offset = torch.cartesian_prod(torch.arange(w), torch.arange(w)).to(points.device) - w//2
        mask = torch.where(img[0] > 0, 1, 0)
        padded_mask = F.pad(mask, (w, w, w, w), mode='constant', value=0)

        windows = points.unsqueeze(1) + offset.unsqueeze(0) + w
        windows = windows.view(-1, 2)
        # bbox = self.generate_bbox_from_point(points, width=w).round().long() + w
        window_feats = padded_mask[windows[:,1], windows[:,0]].view(N, w*w)

        return window_feats


    def forward(self, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            imgs = kwargs.pop('img')
            img_metas = kwargs.pop('img_metas')
            return self.forward_test(imgs, img_metas, **kwargs)

    def random_arange(self, sizes, max_base_size, plus_one=False):

        base_size = max_base_size
        batch_idx_list = []
        batch_size_list = []
        rand_permute = np.random.permutation(len(sizes))
        cur_bin = []
        cur_sizes = []
        cur_size = 0
        for i, idx in enumerate(rand_permute):
            if sizes[idx] < 4:
                continue
            if sizes[idx] + plus_one >= base_size:
                continue
            if cur_size + sizes[idx] + plus_one < base_size:
                cur_bin.append(idx)
                cur_sizes.append(sizes[idx])
                cur_size += sizes[idx] + plus_one
            else:
                # finish the current bin
                batch_idx_list.append(cur_bin)
                batch_size_list.append(cur_sizes)
                cur_bin = [idx]
                cur_size = sizes[idx]
                cur_sizes = [sizes[idx]]

        # deal with the remaining items
        if len(cur_bin) > 0:
            batch_idx_list.append(cur_bin)
            batch_size_list.append(cur_sizes)

        return batch_idx_list, batch_size_list

    def normalize_coordinates(self, graph, ws, type='global'):
        if type == 'global':
            graph = (graph * 2 / ws - 1)
        elif type == 'normalized':
            graph = ((graph + 1) * ws / 2)
            # graph = torch.round(graph).long()
            graph[graph < 0] = 0
            graph[graph >= ws] = ws - 1
        return graph

    def normalize_rings(self, rings, max_length, ori_offsets=None, mask=None, type='global'):

        if type == 'global':
            mask = rings >= 0
            masked_rings = torch.where(mask, rings, 1e8)
            offsets = masked_rings.min(dim=1)[0]

            offset_rings = torch.where(mask, rings - offsets.unsqueeze(1), -1)
            norm_rings = self.normalize_coordinates(offset_rings, max_length, type='global')
            return norm_rings, offsets

        else:
            offset_rings = self.normalize_coordinates(rings, max_length, type='normalized')
            ori_rings = torch.where(mask, offset_rings + ori_offsets.unsqueeze(1), -1)
            return ori_rings, None


    def prepare_node_feats(self, N, batch_idxes, batch_sizes, segments, labels, window_feats, W):
        batch_points = []
        batch_labels = []
        batch_window_feats = []
        device = segments[0].device
        B = len(batch_idxes)

        for i, sizes in enumerate(batch_sizes):
            cur_size = sum(sizes)
            cur_points = torch.zeros(N, 2, device=device)
            cur_labels = torch.zeros(N, device=device, dtype=torch.long) - 1
            cur_window_feats = torch.zeros(N, self.window_feat_size * self.window_feat_size, device=device)

            cur_points[:cur_size] = torch.cat([segments[x] for x in batch_idxes[i]])
            cur_labels[:cur_size] = torch.cat([labels[x] for x in batch_idxes[i]])
            cur_window_feats[:cur_size] = torch.cat([window_feats[x] for x in batch_idxes[i]])

            batch_points.append(cur_points)
            batch_labels.append(cur_labels)
            batch_window_feats.append(cur_window_feats)

        batch_points = torch.stack(batch_points)
        batch_labels = torch.stack(batch_labels)
        batch_window_feats = torch.stack(batch_window_feats)

        attn_mask = ~torch.eye(N, N, device=device, dtype=torch.bool)
        attn_mask = attn_mask.view(1, N, N).repeat(B, 1, 1)

        for idx in range(B):
            start_idx = 0
            for idx2, size in enumerate(batch_sizes[idx]):
                attn_mask[idx, start_idx:start_idx+size, start_idx:start_idx+size] = 0
                start_idx += size

        norm_points = self.normalize_coordinates(batch_points, W, 'global')
        node_feats = torch.cat([batch_points, batch_window_feats], dim=-1)
        node_feats = node_feats.permute(0,2,1).to(device)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.simp_net.num_heads, 1, 1).view(-1, N, N)

        return node_feats, attn_mask, batch_labels

    def apply_noise_window(self, img, contours, w, thre, rand_prob):
        H, W = img.shape
        N = contours.shape[0]
        padded_mask = F.pad(img, (w, w, w, w), mode='constant', value=0)
        offset = torch.cartesian_prod(torch.arange(w), torch.arange(w)).to(img.device) - w//2

        windows = contours.unsqueeze(1) + offset.unsqueeze(0) + w
        windows = windows.view(-1, 2)
        window_feats = padded_mask[windows[:,1], windows[:,0]].view(N, w*w)
        valid_mask = window_feats.sum(dim=1) / (w * w) > thre
        num_valid = valid_mask.sum()

        probs = torch.rand(num_valid, device=img.device)
        bboxes = self.generate_bbox_from_point(contours[valid_mask], width=w)
        keep_idx = mmcv.ops.nms(bboxes, probs, iou_threshold=0.3)[1]
        rand_mask = torch.rand(len(keep_idx)) < rand_prob

        if rand_mask.sum() > 0:
            noise_points = contours[valid_mask][keep_idx[rand_mask]]
            # noise_windows = window_feats[valid_mask][keep_idx[rand_mask]]
            noise_windows = windows.view(N, w*w, 2)[valid_mask][keep_idx[rand_mask]].view(-1, 2)
            noise_state = torch.rand(rand_mask.sum()) < 0.8
            noise_targets = torch.zeros(rand_mask.sum(), w*w, dtype=torch.uint8, device=img.device)
            noise_targets[noise_state] = 1
            padded_mask[noise_windows[:,1], noise_windows[:,0]] = noise_targets.view(-1)

        # return (padded_mask[w:-w, w:-w] - 0.5) / 1.
        return padded_mask[w:-w, w:-w]


    def add_noise(self, img, contours, noise_conf):
        widths = noise_conf['widths']
        thre = noise_conf['cover_thre']
        rand_prob = noise_conf['prob']

        _, H, W = img.shape
        N = contours.shape[0]
        img = (img.clone()[0] > 0).to(torch.uint8)
        contours = torch.tensor(contours, device=img.device).long()

        # crop_img_np = (crop_comp_mask.cpu()[0].permute(1,2,0).numpy()[:,:,0] > 0).astype(np.uint8)
        if noise_conf.get('noise_type', 'iterative'):
            cur_img = img
            for i in range(noise_conf.get('num_iter', 1)):
                w = random.choice(widths)
                cur_contours, _ = cv2.findContours(cur_img.cpu().numpy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if len(cur_contours) > 0:
                    cur_contours = torch.tensor(
                        np.concatenate(cur_contours),
                        device=img.device
                    ).view(-1, 2).long()
                else:
                    cur_contours = torch.zeros(0, 2, device=img.device, dtype=torch.long)
                noise_img = self.apply_noise_window(cur_img, cur_contours, w, thre, rand_prob)
                cur_img = noise_img

        else:
            w = random.choice(widths)
            noise_img = self.apply_noise_window(cur_img, cur_contours, w, thre, rand_prob)

        return (noise_img - 0.5) / 1.0
        # for w in widths:

        padded_mask = F.pad(img, (w, w, w, w), mode='constant', value=0)
        offset = torch.cartesian_prod(torch.arange(w), torch.arange(w)).to(img.device) - w//2

        windows = contours.unsqueeze(1) + offset.unsqueeze(0) + w
        windows = windows.view(-1, 2)
        window_feats = padded_mask[windows[:,1], windows[:,0]].view(N, w*w)
        valid_mask = window_feats.sum(dim=1) / (w * w) > thre
        num_valid = valid_mask.sum()

        probs = torch.rand(num_valid, device=img.device)
        bboxes = self.generate_bbox_from_point(contours[valid_mask], width=w)
        keep_idx = mmcv.ops.nms(bboxes, probs, iou_threshold=0.3)[1]
        rand_mask = torch.rand(len(keep_idx)) < rand_prob

        if rand_mask.sum() > 0:
            noise_points = contours[valid_mask][keep_idx[rand_mask]]
            # noise_windows = window_feats[valid_mask][keep_idx[rand_mask]]
            noise_windows = windows.view(N, w*w, 2)[valid_mask][keep_idx[rand_mask]].view(-1, 2)
            noise_state = torch.rand(rand_mask.sum()) < 0.8
            noise_targets = torch.zeros(rand_mask.sum(), w*w, dtype=torch.uint8, device=img.device)
            noise_targets[noise_state] = 1
            padded_mask[noise_windows[:,1], noise_windows[:,0]] = noise_targets.view(-1)

        return (padded_mask[w:-w, w:-w] - 0.5) / 1.

    def features_to_point_sets(self, features, device):
        rings = []
        ring_start_idx = []

        cnt = 0
        for i, polygons in enumerate(features):
            for j, polygon in enumerate(polygons):
                cur_points = torch.tensor(polygon['exterior'], device=device)
                N = len(cur_points)
                rings.append(cur_points)
                ring_start_idx.append(cnt)
                cnt += N

        return rings, ring_start_idx

    def sample_segments(self, img, rings, ring_idxes, point_labels, N):
        segments = []
        labels = []
        window_feats = []
        for i, ring in enumerate(rings):
            cur_idx = ring_idxes[i]
            ring = ring[:-1]

            if len(ring) <= N:
                cur_segments = ring
                cur_labels = point_labels[cur_idx:cur_idx+len(ring)]
            else:
                cur_point_labels = point_labels[cur_idx:cur_idx+len(ring)]
                start_idx = random.randint(0, len(ring)-1)
                temp = torch.cat([ring[start_idx:], ring[:start_idx]])
                cur_segments = temp[:N]
                cur_labels = torch.cat([cur_point_labels[start_idx:], cur_point_labels[:start_idx]])
                cur_labels = cur_labels[:N]

            cur_window_feats = self.get_window_feats(img, cur_segments.long(), self.window_feat_size)
            segments.append(cur_segments)
            labels.append(cur_labels)
            window_feats.append(cur_window_feats)

        return segments, labels, window_feats

    def polygonize(self, imgs, scale=None):

        B, H, W  = imgs.shape

        polygons = []
        for i in range(B):
            cur_shapes = shapes(imgs[i].cpu().numpy(), mask=imgs[i].cpu().numpy() > 0)

            cur_polygons = {}
            for shape, value in cur_shapes:
                value = int(value)
                if not value in cur_polygons:
                    cur_polygons[value] = []
                # cur_polygons[value].append(shapely.geometry.shape(shape))
                if scale is not None:
                    coords = shape['coordinates']
                    scaled_coords = [(np.array(x) * scale).tolist() for x in coords]
                    shape['coordinates'] = scaled_coords

                cur_polygons[value].append(shape)

            polygons.append(cur_polygons)

        return polygons

    def rasterize(self, batch_features, downscale=4, raster_shape=(256, 256)):

        rasters = []
        for features in batch_features:
            shapes = []
            cnt = 1
            for temp in features:
                for feat in temp:
                    exterior = (np.array(feat['exterior']) / downscale).tolist()
                    interiors = [(np.array(x) / downscale).tolist() for x in feat['interiors']]
                    polygon = geojson.Polygon([exterior], interiors)
                    shapes.append((polygon, cnt))
                cnt += 1

            if len(shapes) > 0:
                raster = rasterio.features.rasterize(
                    shapes, out_shape=raster_shape, dtype=np.int32, all_touched=True
                )
            else:
                raster = np.zeros(raster_shape, dtype=np.int32)

            rasters.append(raster)

        return np.stack(rasters)

    def sample_comp_points(self, mask):
        H, W = mask.shape
        rows, cols = torch.arange(0, H, 8, device=mask.device), torch.arange(0, W, 8, device=mask.device)
        idxes = torch.cartesian_prod(rows, cols)
        valid_mask = mask[idxes[:,1], idxes[:,0]] > 0
        idxes = idxes[valid_mask]

        return idxes

    def sample_grids(self, H, W, stride=8, device='cpu'):
        grid_rows, grid_cols = torch.arange(0, H, stride), torch.arange(0, W, stride)
        grid_idxes = torch.cartesian_prod(grid_rows, grid_cols)
        pixel2grid = torch.zeros(H, W, 4, 2, dtype=torch.long)
        rows, cols = torch.arange(0, H), torch.arange(0, W)
        all_idxes = torch.cartesian_prod(rows, cols)
        pixel2grid[all_idxes[:,0], all_idxes[:,1], 0] = torch.stack([
            all_idxes[:,0] // stride * stride,
            all_idxes[:,1] // stride * stride
        ], dim=1)
        pixel2grid[all_idxes[:,0], all_idxes[:,1], 1] = torch.stack([
            all_idxes[:,0] // stride * stride,
            (all_idxes[:,1] // stride + 1) * stride
        ], dim=1)
        pixel2grid[all_idxes[:,0], all_idxes[:,1], 2] = torch.stack([
            (all_idxes[:,0] // stride + 1) * stride,
            all_idxes[:,1] // stride * stride
        ], dim=1)
        pixel2grid[all_idxes[:,0], all_idxes[:,1], 3] = torch.stack([
            (all_idxes[:,0] // stride + 1) * stride,
            (all_idxes[:,1] // stride + 1) * stride
        ], dim=1)

        return grid_idxes.to(device), pixel2grid.to(device)

    def get_connected_components(self, graph, valid_knn_idxes, valid_mask):
        B, N = valid_mask.shape
        knn_rows = valid_mask.view(-1).nonzero().view(-1)[valid_knn_idxes[:,0]]
        components = []
        for i in range(B):
            temp_mask = (knn_rows >= i * N) & (knn_rows < (i + 1) * N)
            cur_knn_rows = knn_rows[temp_mask] - i * N
            cur_knn_cols = valid_knn_idxes[temp_mask, 1]
            cur_knn_cols = graph[cur_knn_rows, cur_knn_cols]

            G = networkx.Graph()
            G.add_nodes_from(range(N))
            edges = torch.stack([cur_knn_rows, cur_knn_cols], dim=1)
            G.add_edges_from(edges.cpu().numpy())
            cur_comp = connected_components = list(networkx.connected_components(G))
            components.append(cur_comp)

        return components

    def get_pixel_assignment(self, pixel2grid, idxes, mask):
        H, W = mask.shape
        grids = pixel2grid[idxes[:,0], idxes[:,1]]
        assign_mask = torch.zeros(H, W, 4, device=mask.device, dtype=mask.dtype)
        for i in range(4):
            valid_mask = (grids[:,i] < W).all(dim=-1)
            assign_mask[idxes[valid_mask,0], idxes[valid_mask,1], i] = \
                 mask[grids[valid_mask,i,0], grids[valid_mask,i,1]]

        assign_mask = assign_mask.max(dim=-1)[0]
        assign_mask = torch.where(mask.bool(), mask, assign_mask)
        return assign_mask


    def cal_loss_sim(self, raster_gt, feats, K):
        B, _, H, W = raster_gt.size()
        losses, states = {}, {}
        # points = self.sample_comp_points(raster_gt)
        points, pixel2grid = self.sample_grids(H, W, device=raster_gt.device)
        N = points.shape[0]
        points_np = points.cpu().numpy()
        kd_tree = cKDTree(points_np)
        _, graph = kd_tree.query(points_np, K + 1)
        graph = torch.tensor(graph, device=raster_gt.device)
        graph_points = points[graph.view(-1)].unsqueeze(0).repeat(B,1,1).long()
        # graph_points = graph_points.reshape(B*N, 2)

        norm_points = self.normalize_coordinates(points, W, 'global')
        point_feats = F.grid_sample(
            feats,
            norm_points.view(1, 1, -1, 2).repeat(B,1,1,1),
            align_corners=True
        ).squeeze(2).permute(0,2,1) # (B, num_points, num_channels)


        graph_feats = []
        for cur_point_feats in point_feats:
            graph_feats.append(cur_point_feats[graph.view(-1)])

        graph_feats = torch.stack(graph_feats, dim=0).view(B, N, K+1, -1)

        graph_idx = []
        for i in range(B):
            graph_idx.append(raster_gt[i, 0, graph_points[i, :, 1], graph_points[i, :, 0]])
        graph_idx = torch.stack(graph_idx, dim=0).view(B, N, K+1)
        graph_labels = graph_idx[:,:,0:1].repeat(1,1,K+1) == graph_idx

        center_feats = graph_feats[:,:,0:1].repeat(1,1,K+1,1)
        # graph_sim = F.cosine_similarity(center_feats, graph_feats, dim=-1)
        graph_preds = self.linear(torch.cat([center_feats, graph_feats], dim=-1))

        valid_mask = (graph_idx[:,:,0] > 0).view(-1)

        # loss_sim = self.cse_loss_fun(graph_preds.view(B*N, K+1, 2), graph_labels.view(B*N, K+1))
        loss_sim = self.cse_loss_fun(graph_preds.view(-1, 2), graph_labels.view(-1).long())
        loss_sim = loss_sim.view(B*N, -1)[valid_mask].mean()
        losses['loss_sim'] = loss_sim
        states['raster_gt'] = raster_gt[:,0]

        if self.cur_iter % 200 == 0:

            graph_probs = F.softmax(graph_preds.view(-1,K+1,2)[valid_mask], dim=-1)[:,:,1]
            knn_idxes = (graph_probs > 0.5).nonzero()

            comps = self.get_connected_components(graph, knn_idxes, valid_mask.view(B, -1))
            batch_comp_points = []
            batch_comp_labels = []
            batch_pred_masks = []

            for i in range(B):
                comp_points = []
                comp_labels = []
                comp_cnt = 0
                cur_valid_mask = valid_mask.view(B, -1)[i]
                for cur_comp in comps[i]:
                    temp = [x for x in list(cur_comp) if cur_valid_mask[x]]
                    comp_points.append(points[temp])
                    comp_labels.append(torch.tensor([comp_cnt] * len(temp), device=points.device, dtype=torch.long))
                    if len(temp) > 0:
                        comp_cnt += 1

                comp_points = torch.cat(comp_points, dim=0)
                comp_labels = torch.cat(comp_labels, dim=0)

                batch_comp_points.append(comp_points)
                batch_comp_labels.append(comp_labels)
                pred_masks = torch.zeros(H, W, device=img.device, dtype=torch.long)
                pred_masks[comp_points.long()[:,1], comp_points.long()[:,0]] = comp_labels
                temp_idxes = (pred_masks == 0).nonzero()
                pred_masks = self.get_pixel_assignment(
                    pixel2grid, temp_idxes, pred_masks
                )
                pred_masks = torch.where(raster_gt[i,0].bool(), pred_masks, 0)
                batch_pred_masks.append(pred_masks)

            states['comp_points'] = batch_comp_points
            states['comp_labels'] = batch_comp_labels
            states['super_pixel_masks'] = torch.stack(batch_pred_masks, dim=0)

        return losses, states

    def distance_to_segment(self, segment_endpoints, points):
        # Unpack segment endpoints
        x1, y1, x2, y2 = segment_endpoints[:, 0], segment_endpoints[:, 1], segment_endpoints[:, 2], segment_endpoints[:, 3]
        # Unpack points
        px, py = points[:, 0], points[:, 1]

        # Compute the direction vector of the segments
        dx = x2 - x1
        dy = y2 - y1

        # Compute the length of the segments
        segment_lengths = np.sqrt(dx**2 + dy**2)

        # Normalize the direction vector
        dx /= segment_lengths
        dy /= segment_lengths

        # Compute the vector between the first endpoint of each segment and the point
        vx = px - x1
        vy = py - y1

        # Compute the dot product of the vector between the endpoint and the point with the normalized direction vector
        dot_product = vx * dx + vy * dy

        # Find the perpendicular point on the line
        perpendicular_x = x1 + dot_product * dx
        perpendicular_y = y1 + dot_product * dy

        # Calculate distance from point to line segment
        distance_segment = np.sqrt((px - perpendicular_x)**2 + (py - perpendicular_y)**2)

        # Check if perpendicular point falls within the segment bounds
        mask_inside_segment = (dot_product >= 0) & (dot_product <= segment_lengths)

        # Compute distance to closest endpoint if perpendicular projection is outside the segment bounds
        distance_to_endpoints = np.sqrt((px - x1)**2 + (py - y1)**2)
        distance_to_endpoints[mask_inside_segment] = np.sqrt((px - x2)**2 + (py - y2)**2)

        # Final distance is minimum of distance to segment and distance to endpoints
        distances = np.minimum(distance_segment, distance_to_endpoints)

        return distances

    def project_points_onto_segments(self, segments, points):
        # Unpack segment endpoints
        x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
        # Unpack points
        px, py = points[:, 0], points[:, 1]

        # Compute the direction vector of the segments
        dx = x2 - x1
        dy = y2 - y1

        # Compute the length of the segments
        segment_lengths = np.sqrt(dx**2 + dy**2)

        # Normalize the direction vector
        dx /= segment_lengths
        dy /= segment_lengths

        # Compute the vector between the first endpoint of each segment and the point
        vx = px - x1
        vy = py - y1

        # Compute the dot product of the vector between the endpoint and the point with the normalized direction vector
        dot_product = vx * dx + vy * dy

        # Find the perpendicular point on the line
        perpendicular_x = x1 + dot_product * dx
        perpendicular_y = y1 + dot_product * dy

        # Ensure projection falls within segment bounds
        projection_x = np.clip(perpendicular_x, np.minimum(x1, x2), np.maximum(x1, x2))
        projection_y = np.clip(perpendicular_y, np.minimum(y1, y2), np.maximum(y1, y2))

        return np.column_stack((projection_x, projection_y))

    def sample_points_in_ring(self, ring, interval=None):

        interval = self.ring_sample_conf['interval'] if interval is None else interval

        try:
            ring_shape = shapely.LinearRing(ring)
        except ValueError:
            return None

        perimeter = ring_shape.length
        num_bins = max(round(perimeter / interval), 8)
        num_bins = max(num_bins, len(ring))

        bins = np.linspace(0, 1, num_bins)
        sampled_points = [ring_shape.interpolate(x, normalized=True) for x in bins]
        sampled_points = [[temp.x, temp.y] for temp in sampled_points]

        return sampled_points

    def add_noise_to_ring(self, ring, interval=None):
        interval = self.ring_sample_conf['interval'] if interval is None else interval

        noise_type = self.ring_sample_conf.get('noise_type', 'uniform')

        if noise_type == 'random':
            noise_type = random.choice(['uniform', 'skip'])

        if noise_type == 'uniform':
            noise = (np.random.rand(len(ring), 2) - 0.5) * interval / 2.
        elif noise_type == 'skip':
            noise = (np.random.rand(len(ring), 2) - 0.5) * interval
            noise[0:2:-1] = 0

        noisy_ring = ring + torch.tensor(noise)

        return noisy_ring

        if len(sampled_points) > 0:
            sampled_points = (np.array(sampled_points) + noise).tolist()

        return sampled_points

    def get_target_ring(self, ring_A, ring_B):
        sampled_points = self.sample_points_in_ring(ring_A)
        ring_A = torch.tensor(np.array(sampled_points))[:-1]
        ring_B = torch.tensor(np.array(ring_B))[:-1]

        if len(ring_A) < len(ring_B):
            return None, None, None

        assign_result = self.assigner.assign(ring_A, ring_B)

        ring_A_cls_target = torch.zeros(len(ring_A), dtype=torch.long)
        ring_A_reg_target = torch.zeros(len(ring_A), 2)
        segments_A = torch.zeros(len(ring_A), 4)

        temp = assign_result.gt_inds - 1
        temp2 = (temp > -1).nonzero().view(-1)
        ring_A_cls_target[temp2] = 1
        ring_A_reg_target[temp2] = ring_B[temp[temp2]].float()

        ring_A_cls_target.nonzero().view(-1)
        segments_A[:temp2[0]] = torch.cat(
                [ring_A_reg_target[temp2[-1]], ring_A_reg_target[temp2[0]]]
        ).view(1, -1)
        for i, idx in enumerate(temp2):
            left = idx
            right = temp2[i+1] if i < len(temp2) - 1 else len(ring_A)

            seg_x = idx
            seg_y = temp2[i+1] if i < len(temp2) - 1 else temp2[0]

            segments_A[left:right] = torch.cat(
                [ring_A_reg_target[seg_x], ring_A_reg_target[seg_y]]
            ).view(1, -1)

        projs = self.project_points_onto_segments(segments_A[ring_A_cls_target==0], ring_A[ring_A_cls_target==0])
        ring_A_reg_target[ring_A_cls_target==0] = torch.tensor(projs).float()

        return ring_A, ring_A_cls_target, ring_A_reg_target

    def sample_rings(self, batch_gt_polygons, ring_len=20, num_bins=32, pixel_width=3.,
                     scale_factor=4, input_poly_type='noise'):
        batch_rings = []
        batch_ring_cls_targets = []
        batch_ring_reg_targets = []
        batch_ring_angle_targets = []

        for i, gt_polygons in enumerate(batch_gt_polygons):
            rings = []
            ring_cls_targets = []
            ring_reg_targets = []
            ring_angle_targets = []

            for polygon in gt_polygons:
                buffer = 2
                exterior = np.array(polygon['coordinates'][0])
                norm_exterior = (exterior - exterior.min(axis=0, keepdims=True)) / pixel_width + buffer

                """
                sample rings
                """

                if input_poly_type == 'polygonized':
                    shape = (norm_exterior.max(axis=0) + buffer).round().astype(np.int)
                    shape = shape[[1,0]]
                    noisy_norm_exterior = self.add_noise_to_ring(torch.tensor(norm_exterior)).numpy()
                    raster = rasterio.features.rasterize([shapely.Polygon(noisy_norm_exterior)], out_shape=shape, dtype=np.int32, all_touched=True)
                    polygonized = next(shapes(raster, mask=raster > 0))
                    ring = polygonized[0]['coordinates'][0]
                    ring, cls_target, reg_target = self.get_target_ring(ring, norm_exterior)

                    if ring is None:
                        continue

                    ring = ring.round().to(torch.int)

                else:
                    # ring = np.array(self.sample_points_in_ring(norm_exterior, add_noise=True)).astype(np.int)
                    ring = self.sample_points_in_ring(norm_exterior)
                    if ring is None:
                        continue

                    ring = np.array(ring)
                    ring, cls_target, reg_target = self.get_target_ring(ring, norm_exterior)
                    ring = self.add_noise_to_ring(ring)
                    if scale_factor > 1:
                        ring = ring // scale_factor * scale_factor
                    ring = ring.to(torch.int)

                start_idx = random.randint(0, len(ring)-1)
                shuffled_ring = torch.cat([ring[start_idx:], ring[:start_idx]])
                shuffled_cls_target = torch.cat([cls_target[start_idx:], cls_target[:start_idx]])
                shuffled_reg_target = torch.cat([reg_target[start_idx:], reg_target[:start_idx]])

                ext_polygon = torch.cat([shuffled_reg_target, shuffled_reg_target[0:1]])
                diff = ext_polygon[1:] - ext_polygon[:-1]
                degs = torch.fmod(torch.arctan2(- diff[:,1], diff[:,0]), 2 * np.pi)
                bins = torch.linspace(0, 2 * math.pi, num_bins + 1)
                bin_indices = torch.searchsorted(bins, degs, right=True)  # right closed
                shuffled_angle_target = torch.eye(num_bins)[bin_indices - 1]

                sampled_ring = torch.zeros(ring_len, 2, dtype=shuffled_ring.dtype) - 1
                sampled_cls_target = torch.zeros(ring_len, dtype=shuffled_cls_target.dtype)
                sampled_reg_target = torch.zeros(ring_len, 2, dtype=shuffled_reg_target.dtype) - 1
                sampled_angle_target = torch.zeros(ring_len, num_bins, dtype=shuffled_reg_target.dtype) - 1

                sampled_ring[:len(shuffled_ring)] = shuffled_ring[:ring_len]
                sampled_cls_target[:len(shuffled_ring)] = shuffled_cls_target[:ring_len]
                sampled_reg_target[:len(shuffled_ring)] = shuffled_reg_target[:ring_len]
                sampled_angle_target[:len(shuffled_ring)] = shuffled_angle_target[:ring_len]

                if sampled_cls_target.sum() >= 1:
                    rings.append(sampled_ring)
                    ring_cls_targets.append(sampled_cls_target)
                    ring_reg_targets.append(sampled_reg_target)
                    ring_angle_targets.append(sampled_angle_target)


            if len(rings) > 0:
                num_max_ring = self.ring_sample_conf.get('num_max_ring', 512)
                batch_rings.append(torch.stack(rings)[:num_max_ring])
                batch_ring_cls_targets.append(torch.stack(ring_cls_targets)[:num_max_ring])
                batch_ring_reg_targets.append(torch.stack(ring_reg_targets)[:num_max_ring])
                batch_ring_angle_targets.append(torch.stack(ring_angle_targets)[:num_max_ring])
            else:
                batch_rings.append(torch.ones(1, ring_len, 2)) # dummy points
                batch_ring_cls_targets.append(torch.zeros(1, ring_len, dtype=torch.long))
                batch_ring_reg_targets.append(torch.ones(1, ring_len, 2))
                batch_ring_angle_targets.append(torch.ones(1, ring_len, num_bins))

        return batch_rings, batch_ring_cls_targets, batch_ring_reg_targets, batch_ring_angle_targets

    def cal_loss_angle(self, pred, target, mask, cls_target=None, type='all_points'):

        def angle_between_segments(segments):
            # Unpack the segments
            p1, p2, p3 = segments[:, 0], segments[:, 1], segments[:, 2]

            # Compute vectors representing the segments
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate dot product and magnitudes
            dot_product = torch.sum(v1 * v2, dim=1)
            magnitude_v1 = torch.norm(v1, dim=1)
            magnitude_v2 = torch.norm(v2, dim=1)

            valid_mask = (magnitude_v1 > 0) & (magnitude_v2 > 0)

            # Calculate the angle between the vectors
            cosine_angle = dot_product / (magnitude_v1 * magnitude_v2 + 1e-8)
            cosine_angle = torch.clip(cosine_angle, -1, 1)
            angle_radians = torch.acos(cosine_angle)

            # Convert radians to degrees
            # angle_degrees = angle_radians * (180.0 / torch.pi)

            return angle_radians, valid_mask

        def get_segments(points):
            A = points[:,:-2]
            B = points[:,1:-1]
            C = points[:,2:]
            return torch.stack([A,B,C], dim=2)

        B, N, _ = pred.shape

        if type == 'all_points':
            pred_segments = get_segments(pred)
            target_segments = get_segments(target)
            mask = get_segments(mask).view(B, N-2, -1).all(dim=-1)

            pred_angles, pred_valid_mask = angle_between_segments(pred_segments.view(-1, 3, 2))
            target_angles, target_valid_mask = angle_between_segments(target_segments.view(-1, 3, 2))
            valid_mask = pred_valid_mask & target_valid_mask & mask.view(-1)
            loss = self.l1_loss_fun(pred_angles[valid_mask], target_angles[valid_mask]).mean()
        elif type == 'gt_points':
            losses = []
            for i in range(B):
                cur_pred = pred[i, cls_target[i] > 0]
                cur_target = target[i, cls_target[i] > 0]

                if len(cur_target) < 3:
                    continue

                cur_pred_segments = get_segments(cur_pred.unsqueeze(0))
                cur_target_segments = get_segments(cur_target.unsqueeze(0))

                cur_pred_angles, cur_pred_valid_mask = angle_between_segments(cur_pred_segments.view(-1, 3, 2))
                cur_target_angles, cur_target_valid_mask = angle_between_segments(cur_target_segments.view(-1, 3, 2))
                valid_mask = cur_pred_valid_mask & cur_target_valid_mask

                if valid_mask.sum() > 0:
                    cur_loss = self.l1_loss_fun(cur_pred_angles[valid_mask], cur_target_angles[valid_mask]).mean()
                    if torch.isnan(cur_loss):
                        pdb.set_trace()
                    losses.append(cur_loss)

            loss = torch.tensor(losses).mean() if len(losses) > 0 else torch.zeros(1, device=pred.device)

        return {'loss_ring_angle': loss * self.loss_weights['ring_angle']}, {}

    def cal_loss_ring_next(self, ring_feats, ring_cls_targets, valid_mask):
        N, C, L = ring_feats.shape
        scores = self.score_net(ring_feats)
        gt_scores = torch.zeros(N, L, dtype=torch.long, device=ring_feats.device) - 1
        for i in range(N):
            cur_len = valid_mask[i].sum()
            gt_idxes = ring_cls_targets[i].nonzero().view(-1)
            start_idx = 0
            for idx in gt_idxes:
                gt_scores[i, start_idx:idx] = idx
                start_idx = idx
        score_mask = (gt_scores >= 0).view(-1)
        loss = torch.zeros(1, device=ring_feats.device)
        if score_mask.sum() > 0:
            loss = self.cse_loss_fun(scores.view(N*L, -1)[score_mask], gt_scores.view(N*L)[score_mask]).mean()
        if torch.isnan(loss):
            pdb.set_trace()

        return {'loss_ring_next': loss * self.loss_weights['ring_next']}, {'pred_next': scores}



    def forward_train(self, gt_features, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsipoly/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        states = dict()

        rings, ring_cls_targets, ring_reg_targets, ring_angle_targets = self.sample_rings(
            gt_features, ring_len=self.ring_sample_conf['length'],
            input_poly_type=self.train_cfg.get('input_poly_type', 'noise')
        )


        """
        gdal_features = [x['features'] for x in gdal_features]
        gt_features = [x['features'] for x in gt_features]
        h_crop, w_crop = self.train_cfg.crop_size
        assert h_crop == w_crop
        B, _, H, W = img.size()
        K = 4
        # assert B == 1
        # contours = torch.tensor(contours[0], device=img.device)
        # gt_points = torch.tensor(gt_points[0], device=img.device, dtype=torch.float)
        # gt_degrees = torch.tensor(gt_degrees[0], device=img.device, dtype=torch.float)
        # contour_labels = torch.tensor(contour_labels[0], device=img.device)
        # polygons = polygons[0]
        # num_degrees = gt_degrees.shape[-1]

        raster_gt = self.rasterize(gt_features)
        raster_gt = torch.tensor(raster_gt, device=img.device).float()
        # raster_gt = F.interpolate(raster_gt.unsqueeze(1), size=(H, W)).int()

        img = F.interpolate(raster_gt.unsqueeze(1), size=(H, W)).int()
        img = (img > 0).float() - 0.5
        img = img.repeat(1,3,1,1)
        """

        """
        if self.noise_conf is not None:
            noise_imgs = []
            for cur_img, cur_contours in zip(img, contours):
                cur_img_np = (cur_img[0].cpu().numpy() > 0).astype(np.uint8)
                cur_contours, _ = cv2.findContours(cur_img_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if len(cur_contours) > 0:
                    cur_contours = np.concatenate(cur_contours, axis=0).reshape(-1, 2)
                else:
                    cur_contours = np.zeros((0,2), dtype=np.int32)
                cur_contours = torch.tensor(cur_contours, device=img.device)
                cur_noise_img = self.add_noise(cur_img, cur_contours, self.noise_conf)
                noise_imgs.append(cur_noise_img)

            noise_imgs = torch.stack(noise_imgs, dim=0)
            noise_imgs = noise_imgs.unsqueeze(1).repeat(1,3,1,1)
            x = self.extract_feat(noise_imgs)
            states['noise_imgs'] = noise_imgs
        else:
            x = self.extract_feat(img)

        loss_decode, state_decode = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg
        )
        feats = state_decode['feats']
        # seg_logits = state['seg_logits']
        losses.update(loss_decode)
        states.update(state_decode)

        loss_sim = self.cal_loss_sim(raster_gt, feats, K)
        """

        # polygons = self.polygonize(raster_gt, scale=4)

        # rings, ring_cls_targets, ring_reg_targets, ring_angle_targets = self.sample_rings(
        #     polygons, gt_features, ring_len=self.ring_sample_conf['length']
        # )

        batch_sizes = [len(x) for x in rings]


        device = (self.backbone.parameters()).__next__().device

        rings = torch.cat(rings, dim=0).to(device)
        ring_cls_targets = torch.cat(ring_cls_targets, dim=0).to(device)
        ring_reg_targets = torch.cat(ring_reg_targets, dim=0).to(device)
        ring_angle_targets = torch.cat(ring_angle_targets, dim=0).to(device).argmax(dim=-1)

        max_sample_length = self.ring_sample_conf['interval'] * self.ring_sample_conf['length']
        norm_rings, offset_rings = self.normalize_rings(
            rings, max_sample_length, 'global'
        )
        norm_ring_reg_targets, offset_ring_reg_targets = self.normalize_rings(
            ring_reg_targets, max_sample_length, 'global'
        )

        N, L, _ = rings.shape
        seq = (torch.arange(L) / (L - 1) * 2 - 1).view(1, -1, 1).repeat(N,1,1).to(device)
        rings_input = torch.cat([norm_rings, seq], dim=-1)

        ring_feats = self.simp_net(rings_input.permute(0,2,1).to(device).float())
        ring_pred_cls = self.ring_cls_net(ring_feats.permute(0,2,1))
        ring_pred_reg = self.ring_reg_net(ring_feats.permute(0,2,1))
        ring_pred_angle = self.ring_angle_net(ring_feats.permute(0,2,1))

        max_offset = self.ring_sample_conf.get('max_offset', -1)
        if max_offset > 0:
            # ring_pred_reg = ring_pred_reg 
            ring_pred_reg = (max_offset / max_sample_length) * ring_pred_reg

        loss_ring_cls = self.cse_loss_fun(
            ring_pred_cls.view(-1,2), ring_cls_targets.view(-1)
        )
        loss_ring_reg = self.l1_loss_fun(
            ring_pred_reg.view(-1,2), norm_ring_reg_targets.view(-1,2) - norm_rings.view(-1,2)
        )
        loss_ring_angle = self.cse_loss_fun(
            ring_pred_angle.view(-1,32), ring_angle_targets.view(-1)
        )
        mask = rings >= 0


        if 'ring_angle' in self.loss_weights:
            loss_ring_ang, _ = self.cal_loss_angle(
                norm_rings + ring_pred_reg, norm_ring_reg_targets, mask, ring_cls_targets,
                type=self.train_cfg.get('angle_type', 'all_points')
            )
            losses.update(loss_ring_ang)

        if 'ring_next' in self.loss_weights:
            loss_ring_next, state_next = self.cal_loss_ring_next(ring_feats, ring_cls_targets, mask.all(dim=-1))
            next_scores = state_next['pred_next']
            losses.update(loss_ring_next)

        loss_ring_cls = loss_ring_cls[mask.all(dim=-1).view(-1)].mean()
        loss_ring_reg = loss_ring_reg.view(-1)[mask.view(-1)].mean()
        loss_ring_angle = loss_ring_angle[mask.all(dim=-1).view(-1)].mean()

        losses['loss_ring_cls'] = loss_ring_cls * self.loss_weights['ring_cls']
        losses['loss_ring_reg'] = loss_ring_reg * self.loss_weights['ring_reg']
        losses['loss_ring_pred_angle'] = loss_ring_angle * self.loss_weights['ring_pred_angle']


        pred_rings, _ = self.normalize_rings(
            norm_rings + ring_pred_reg, self.ring_sample_conf['interval'] * self.ring_sample_conf['length'],
            ori_offsets=offset_rings, type='normalized', mask=mask
        )

        batch_pred_rings = []
        batch_gt_rings = []
        batch_next_rings = []
        start_idx = 0
        offset_x, offset_y = 0, 0
        height, width = 1024, 1024
        # step_size = self.ring_sample_conf['interval'] * self.ring_sample_conf['length']
        step_size = 100

        for size in batch_sizes:
            temp = pred_rings[start_idx:start_idx+size].cpu()
            temp2 = next_scores[start_idx:start_idx+size].cpu()

            cur_rings = []
            cur_gt_rings = []
            cur_next_rings = []
            for ring, gt_ring, cur_next_scores in zip(temp, ring_reg_targets[start_idx:start_idx+size].cpu(), temp2):
                cur_ring = ring[(ring >= 0).all(dim=-1)].view(-1, 2)
                cur_gt_ring = gt_ring[(gt_ring >= 0).all(dim=-1)].view(-1, 2)
                cur_offset = torch.tensor([offset_x, offset_y]).view(1,2)

                cur_rings.append(cur_ring + cur_offset)
                cur_gt_rings.append(cur_gt_ring + cur_offset)

                next_idxes = cur_next_scores.max(dim=1)[1]
                x = 0
                temp = []
                while(next_idxes[x] > x and (ring >= 0).all(dim=1)[x]):
                    temp.append(x)
                    x = next_idxes[x]

                if next_idxes[x] > x and (ring >= 0).all(dim=1)[x]:
                    temp.append(x)

                temp = torch.tensor(temp).long()
                cur_next_ring = ring[temp]

                cur_next_rings.append(cur_next_ring + cur_offset)

                if offset_x + step_size >= width:
                    if offset_y + step_size < height:
                        offset_x = 0
                        offset_y += step_size
                    else:
                        break
                else:
                    offset_x += step_size



            batch_pred_rings.append(cur_rings)
            batch_gt_rings.append(cur_gt_rings)
            batch_next_rings.append(cur_next_rings)

            start_idx += size

        states['pred_rings'] = batch_pred_rings
        states['gt_rings'] = batch_gt_rings
        states['next_rings'] = batch_next_rings


        return losses, states




        gdal_rings, gdal_ring_idxes = self.features_to_point_sets(gdal_features, device=img.device)
        gt_rings, gt_ring_idxes = self.features_to_point_sets(gt_features, device=img.device)

        gdal_points = torch.cat(gdal_rings)
        gt_points = torch.cat(gt_rings)

        # if len(gdal_points) < len(gt_points):
        #     losses['loss_rings'] = torch.zeros(1, device=img.device)
        #     return losses, states

        gdal_point_labels = torch.zeros(len(gdal_points), device=gdal_points.device, dtype=torch.long)

        assign_result = self.assigner.assign(
            gdal_points, gt_points, img_metas[0]
        )

        temp = assign_result.gt_inds - 1
        temp2 = (temp > -1).nonzero().view(-1)
        gdal_point_labels[temp2] = 1

        segments, labels, window_feats = self.sample_segments(
            img[0], gdal_rings, gdal_ring_idxes, gdal_point_labels, self.len_sampled_segments,
        )
        sizes = [len(x) for x in segments]
        batch_idxes, batch_sizes = self.random_arange(sizes, self.len_sampled_segments, False)
        num_limit_points = 512 * 512 * 32
        num_max_batch = num_limit_points // self.len_sampled_segments ** 2

        ## TODO: shuffle the list
        if len(batch_idxes) > num_max_batch:
            batch_idxes = batch_idxes[:num_max_batch]
            batch_sizes = batch_sizes[:num_max_batch]

        node_feats, attn_mask, node_labels = self.prepare_node_feats(
            self.len_sampled_segments, batch_idxes, batch_sizes, segments, labels, window_feats, W
        )
        node_preds = self.simp_net(node_feats, attn_mask)
        node_preds = node_preds.permute(0,2,1).contiguous()
        mask = (node_labels >= 0).view(-1)
        loss = self.cse_loss_fun(node_preds.view(-1, 2)[mask], node_labels.view(-1)[mask])

        losses['loss_rings'] = loss
        states['gdal_polygons'] = [gdal_rings]
        states['gt_polygons'] = [gt_rings]

        return losses, states

    def get_crop_boxes(self, img_H, img_W, crop_size=256, stride=192):
        # prepare locations to crop
        num_rows = math.ceil((img_H - crop_size) / stride) if math.ceil(
            (img_H - crop_size) /
            stride) * stride + crop_size >= img_H else math.ceil(
                (img_H - crop_size) / stride) + 1
        num_cols = math.ceil((img_W - crop_size) / stride) if math.ceil(
            (img_W - crop_size) /
            stride) * stride + crop_size >= img_W else math.ceil(
                (img_W - crop_size) / stride) + 1

        x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
        xmin = x * stride
        ymin = y * stride

        xmin = xmin.ravel()
        ymin = ymin.ravel()
        xmin_offset = np.where(xmin + crop_size > img_W, img_W - xmin - crop_size,
                               np.zeros_like(xmin))
        ymin_offset = np.where(ymin + crop_size > img_H, img_H - ymin - crop_size,
                               np.zeros_like(ymin))
        boxes = np.stack([
            xmin + xmin_offset, ymin + ymin_offset,
            np.minimum(xmin + crop_size, img_W),
            np.minimum(ymin + crop_size, img_H)
        ], axis=1)

        return boxes

    def init_comp_buffer(self, contours, contour_labels, crop_boxes):
        comp_idxes = contour_labels.unique()
        num_patch_per_comp = np.zeros((len(comp_idxes)+1,), dtype=np.int)
        cur_num_patch_per_comp = np.zeros((len(comp_idxes)+1,), dtype=np.int)

        # num_points_per_comp = np.zeros((len(comp_idxes)+1,), dtype=np.int)
        # for idx in comp_idxes:
        #     num_points_per_comp[idx] = (contour_labels == idx).sum()
        for crop_idx, crop_box in enumerate(crop_boxes):
            start_x, start_y, end_x, end_y = crop_box
            within_bbox_mask = \
                    (contours[:, 0] >= start_x) & \
                    (contours[:, 0] < end_x) & \
                    (contours[:, 1] >= start_y) & \
                    (contours[:, 1] < end_y)
            comp_idx = contour_labels[within_bbox_mask].unique().cpu().numpy()
            if len(comp_idx) > 0:
                num_patch_per_comp[comp_idx] += 1

        # self.num_points_per_comp = num_points_per_comp
        self.num_patch_per_comp = num_patch_per_comp
        self.cur_num_patch_per_comp = cur_num_patch_per_comp
        self.comp_idxes = comp_idxes
        self.point_feats_buffer = {}
        self.point_preds_buffer = {}
        self.point_pool = []

    def separate_ring(self, ring, crop_len, stride):

        N, _ = ring.shape
        if N < crop_len:
            ring = np.concatenate([ring, np.zeros((crop_len-N, 2)) - 1])
            return [ring]

        repeated_ring = np.concatenate([ring[:-1], ring], axis=0)

        num_parts = math.ceil((N - crop_len) / stride) \
                if math.ceil((N - crop_len) / stride) * stride + crop_len >= N \
                else math.ceil((N - crop_len) / stride) + 1

        idxes = np.arange(num_parts + 1)  * stride
        # offset = np.where(idxes + crop_len > N, N - crop_len, idxes)

        rings = [repeated_ring[x:x + crop_len] for x in idxes]
        return rings

    def decode_ring_next(self, points, next_idxes, pred_angles, num_angles, valid_mask=None):
        x = 0
        pred_idxes = []
        while(next_idxes[x] > x and (valid_mask is None or valid_mask[x])):
            pred_idxes.append(x)
            x = next_idxes[x]

        if valid_mask is None or valid_mask[x]:
            pred_idxes.append(x)

        pred_idxes = torch.tensor(pred_idxes).long()
        next_angles = pred_angles[pred_idxes]

        # pred_points = points[pred_idxes]
        # ext_polygon = torch.cat([pred_points, pred_points[0:1]])
        # diff = ext_polygon[1:] - ext_polygon[:-1]
        # degs = torch.fmod(torch.arctan2(- diff[:,1], diff[:,0]), 2 * np.pi)

        # bins = torch.linspace(0, 2 * math.pi, num_bins + 1)
        # bin_indices = torch.searchsorted(bins, degs, right=True)  # right closed
        # shuffled_angle_target = torch.eye(num_bins)[bin_indices - 1]


        return pred_idxes

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, contours=None, contour_labels=None,
                        comp_mask=None, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        interval = self.test_cfg['interval']
        length = self.test_cfg['length']
        num_max_ring = self.test_cfg['num_max_ring']
        ring_stride = self.test_cfg['ring_stride']
        num_angles = self.ring_sample_conf.get('num_angles', 32)

        poly_json = self.polygonize((img[:,0] > 0).int())
        polygons = [x['coordinates'] for x in poly_json[0][1]]

        all_rings = []
        all_ring_sizes = []
        all_idxes = []
        for i, rings in tqdm(enumerate(polygons)):
            sizes = []
            for j, ring in enumerate(rings):
                sampled_ring = np.array(self.sample_points_in_ring(ring, interval=interval))
                ring_parts = np.array(self.separate_ring(sampled_ring, crop_len=length, stride=ring_stride))
                idx1 = np.array([i] * len(ring_parts))
                idx2 = np.array([j] * len(ring_parts))
                idx = np.stack([idx1, idx2], axis=1)
                all_rings.append(ring_parts)
                all_idxes.append(idx)
                sizes.append(len(sampled_ring))

            all_ring_sizes.append(sizes)

            # if i == 1000:
            #     break

        all_rings = np.concatenate(all_rings, axis=0)
        all_idxes = np.concatenate(all_idxes, axis=0)
        all_rings = torch.tensor(all_rings)
        all_idxes = torch.tensor(all_idxes)

        N, L, _ = all_rings.shape
        norm_rings, offset_rings = self.normalize_rings(
            all_rings, interval * length, 'global'
        )

        seq = (torch.arange(L) / (L - 1) * 2 - 1).view(1, -1, 1).repeat(N,1,1)
        rings_input = torch.cat([norm_rings, seq], dim=-1)

        num_iter = N // num_max_ring if N % num_max_ring == 0 else N // num_max_ring + 1
        start_idx = 0
        ring_pred_cls = []
        ring_pred_reg = []
        ring_pred_next = []
        ring_pred_angle = []
        for i in tqdm(range(num_iter)):
            temp = rings_input[start_idx:start_idx+num_max_ring].permute(0,2,1).float()
            ring_feats = self.simp_net(temp.to(img.device))
            cur_ring_pred_cls = self.ring_cls_net(ring_feats.permute(0,2,1)).cpu()
            cur_ring_pred_reg = self.ring_reg_net(ring_feats.permute(0,2,1)).cpu()
            cur_ring_pred_angle = self.ring_angle_net(ring_feats.permute(0,2,1)).cpu()
            if self.test_cfg.get('use_ring_next', False):
                cur_ring_pred_next = self.score_net(ring_feats).cpu()
                ring_pred_next.append(cur_ring_pred_next)

            ring_pred_cls.append(cur_ring_pred_cls)
            ring_pred_reg.append(cur_ring_pred_reg)
            ring_pred_angle.append(cur_ring_pred_angle)
            start_idx += num_max_ring

        ring_pred_cls = torch.cat(ring_pred_cls, dim=0)
        ring_pred_reg = torch.cat(ring_pred_reg, dim=0)
        ring_pred_next = torch.cat(ring_pred_next, dim=0)
        ring_pred_angle = torch.cat(ring_pred_angle, dim=0)

        # ring_feats = self.simp_net(rings_input.permute(0,2,1).to(img.device).float())
        # ring_pred_cls = self.ring_cls_net(ring_feats.permute(0,2,1))
        # ring_pred_reg = self.ring_reg_net(ring_feats.permute(0,2,1))


        max_offset = self.ring_sample_conf.get('max_offset', -1)
        if max_offset > 0:
            max_sample_length = self.ring_sample_conf['interval'] * self.ring_sample_conf['length']
            ring_pred_reg = (max_offset / max_sample_length) * ring_pred_reg

        mask = all_rings >= 0
        pred_rings, _ = self.normalize_rings(
            norm_rings + ring_pred_reg, interval * length,
            ori_offsets=offset_rings, type='normalized', mask=mask
        )
        pred_rings = pred_rings.cpu()

        pred_polygons = []
        for i in range(len(all_ring_sizes)):
            cur_pred_polygon = []
            for j in range(len(all_ring_sizes[i])):
                cur_mask = (all_idxes[:,0] == i) & (all_idxes[:,1] == j)
                cur_rings = pred_rings[cur_mask]
                cur_ring_len = all_ring_sizes[i][j]
                cur_ring_pred_angle = ring_pred_angle[cur_mask]

                cur_pred_ring = torch.zeros(cur_ring_len, 2)
                cur_pred_angle = torch.zeros(cur_ring_len, num_angles)
                cur_count = torch.zeros(cur_ring_len)

                if self.test_cfg.get('use_ring_next', False):
                    cur_ring_next = ring_pred_next[cur_mask]
                    ring_next = torch.zeros(cur_ring_len, cur_ring_len)

                for k in range(cur_rings.shape[0]):
                    cur_valid_mask = (cur_rings[k] >= 0).all(dim=1)
                    temp = (torch.arange(length) + ring_stride * k) % cur_ring_len
                    cur_pred_ring[temp[cur_valid_mask]] += cur_rings[k][cur_valid_mask]
                    cur_pred_angle[temp[cur_valid_mask]] += cur_ring_pred_angle[k][cur_valid_mask]
                    cur_count[temp] += 1
                    if self.test_cfg.get('use_ring_next', False):
                        ring_next[temp[cur_valid_mask,None], temp[cur_valid_mask]] += cur_ring_next[k, cur_valid_mask][:, cur_valid_mask]

                temp = cur_pred_ring[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)

                if self.test_cfg.get('use_ring_next', False):
                    pred_idxes = self.decode_ring_next(temp, ring_next.max(dim=1)[1], cur_pred_angle.max(dim=-1)[1], num_angles)
                    cur_pred_polygon.append(temp[pred_idxes])
                else:
                    cur_pred_polygon.append(temp)

            pred_polygons.append(cur_pred_polygon)

        return pred_polygons

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit, state = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                # mode='bilinear',
                mode='bicubic',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit, state

    def inference(self, img, img_meta, rescale, contours=None, contour_labels=None, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsipoly/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        if isinstance(img_meta, DataContainer):
            img_meta = img_meta.data
            img_meta = img_meta[0]

        result = {}

        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            pred_polygons = self.slide_inference(
                img, img_meta, rescale, contours=contours, contour_labels=contour_labels, **kwargs
            )
            result['polygons_v2'] = pred_polygons
            """
            result['vis|polygons'] = [img, polygons]
            result['vis|points_corners'] = [img, contours]
            result['vis|points_vertices'] = [img, vertices]
            result['polygons'] = polygons
            """
            # result['polygons'] = kwargs['polygons'][0]
        else:
            seg_logit, state = self.whole_inference(img, img_meta, rescale, **kwargs)
            feats = state['feats']
            contours = [corner_point.view(-1, 2) for corner_point in contours]
            polygons, vertices = self.get_polygons(img, seg_logit, feats, contours)
            probs = F.softmax(seg_logit, dim=1)[:, 1:, ...]
            result = dict(
                probs=probs,
                vertices=vertices,
                polygons=polygons
            )
            # flip = img_meta[0]['flip']
            # if flip:
            #     flip_direction = img_meta[0]['flip_direction']
            #     assert flip_direction in ['horizontal', 'vertical']
            #     if flip_direction == 'horizontal':
            #         probs = probs.flip(dims=(3, ))
            #     elif flip_direction == 'vertical':
            #         probs = probs.flip(dims=(2, ))

            result['vis|polygons'] = [img, polygons]
            result['vis|featmap'] = [img, F.softmax(seg_logit, dim=1)[:, 1:, ...]]
            result['vis|featmap_contour'] = [img, probs]
            result['vis|points_corners'] = [img, contours]
            result['vis|points_vertices'] = [img, vertices]

        return result


    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        results = self.inference(img, img_meta, rescale, **kwargs)
        # probs = results['probs']
        # seg_pred = probs.argmax(dim=1)
        # if torch.onnx.is_in_onnx_export():
        #     # our inference backend only support 4D output
        #     seg_pred = seg_pred.unsqueeze(0)
        #     return seg_pred
        # seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        # seg_pred = list(seg_pred)

        # results['pred'] = seg_pred
        return results

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


