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


@SEGMENTORS.register_module()
class PolyRegularizerV4(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self, backbone, decode_head, discriminator, gan_loss_conf, assigner, sampler, wandb_cfg=None,
        window_feat_size=6, len_sampled_segments=50, noise_conf=None, gan_patch_size=128,
        dice_loss_conf=None, ring_sample_conf=None, **kwargs,
    ):
        super(PolyRegularizerV4, self).__init__(backbone, decode_head, **kwargs)
        self.backbone = builder.build_backbone(backbone)
        # self.simp_net = builder.build_backbone(backbone)
        self.discriminator = builder.build_gan_module(discriminator)
        self.gan_loss = builder.build_loss(gan_loss_conf)
        self.gan_loss_conf = gan_loss_conf
        self.gan_patch_size = gan_patch_size

        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler)
        self.wandb_cfg = wandb_cfg
        self.nms_width = 7
        self.cur_iter = 0
        self.disc_step = self.train_cfg.get('disc_step', 5) if self.train_cfg is not None else -1
        self.window_feat_size = window_feat_size
        self.len_sampled_segments = len_sampled_segments
        self.cse_loss_fun = nn.CrossEntropyLoss()
        self.dice_loss_fun = None
        if dice_loss_conf:
            self.dice_loss_fun = builder.build_loss(dice_loss_conf)
        self.noise_conf = noise_conf
        self.ring_sample_conf = ring_sample_conf

        self.generator = nn.ModuleList([self.backbone, self.decode_head])

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

    def cal_gan_loss(self, img, logits, is_disc=False):
        B, C, H, W = img.shape
        patch_size = self.gan_patch_size
        assert patch_size <= H and patch_size <= W
        # offset_x, offset_y = torch.randint(0, H-patch_size), torch.randint(0, W-patch_size)
        offset = torch.randint(0, H-patch_size, size=(2,))
        up_logits = F.interpolate(logits, size=(H, W), mode='nearest')

        true_img = img[:, 0:1, offset[1]:offset[1]+patch_size, offset[0]:offset[0]+patch_size]
        fake_img = up_logits[:, 1:2, offset[1]:offset[1]+patch_size, offset[0]:offset[0]+patch_size]

        if not is_disc:
            with torch.no_grad():
                true_preds = self.discriminator(true_img)
                fake_preds = self.discriminator(fake_img)
        else:
            true_preds = self.discriminator(true_img)
            fake_preds = self.discriminator(fake_img)

        disc_loss = self.gan_loss(true_preds, True, is_disc)
        disc_loss += self.gan_loss(fake_preds, False, is_disc)

        return {'gan_loss': disc_loss / 2.0}

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
        # img = data_batch['img']
        # B, C, H, W = img.shape
        self.cur_iter += 1

        if self.disc_step > 0 and self.cur_iter % self.disc_step == 1:

            optimizer['discriminator'].zero_grad()

            with torch.no_grad():
                losses, states = self(**data_batch)

            seg_logits = states['seg_logits']
            raster_gt = states['raster_gt']

            if self.gan_loss_conf['loss_weight'] > 0:
                loss_gan = self.cal_gan_loss(raster_gt - 0.5, seg_logits, is_disc=True)
                losses.update(loss_gan)

            loss, log_vars = self._parse_losses(losses)
            loss.backward()
            optimizer['discriminator'].step()

        else:

            optimizer['generator'].zero_grad()
            losses, states = self(**data_batch)
            seg_logits = states['seg_logits']
            raster_gt = states['raster_gt']

            if self.gan_loss_conf['loss_weight'] > 0:
                loss_gan = self.cal_gan_loss(raster_gt - 0.5, seg_logits, is_disc=True)
                losses.update(loss_gan)
                loss_gan = self.cal_gan_loss(raster_gt - 0.5, seg_logits, is_disc=False)

            loss, log_vars = self._parse_losses(losses)
            loss.backward()
            optimizer['generator'].step()

        img = states['noise_imgs']
        # states['vis|masks_gt'] = [data_batch['img'], data_batch['gt_semantic_seg'][:, 0].cpu().numpy()]
        states['vis|masks_noise_img_gt'] = [img, states['raster_gt'][:, 0].cpu().numpy()]

        if 'seg_logits' in states:
            probs = F.softmax(states['seg_logits'], dim=1)[:, 1:, ...].detach()
            states['vis|featmap_probs'] = [img, probs]
            # states['vis|points_vertices'] = [data_batch['img'], data_batch['contours']]
            # states['vis|points_pos_point_preds'] = [data_batch['img'], states['pos_point_preds']]

        if 'point_preds_ori' in states:
            point_preds = states['point_preds_ori'].detach()
            states['vis|points_point_preds'] = [img, point_preds.view(1, -1, 2)]

        if 'gdal_polygons' in states:
            polygons = states['gdal_polygons']
            states['vis|polygons_gdal_polygons'] = [img, [polygons]]

        if 'gt_polygons' in states:
            polygons = states['gt_polygons']
            states['vis|polygons_gt_polygons'] = [img, [polygons]]

        if 'pred_bf' in states:
            states['vis|masks_pred_bf'] = [states['pred_bf'], states['raster_gt'][:, 0].cpu().numpy()]


        """
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states
        )
        """

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states
        )

        return outputs

    def get_within_bbox_rings(self, rings, start_x, start_y, end_x, end_y):
        idxes = []
        for i, ring in enumerate(rings):
            within_bbox_mask = \
                    (ring[:, 0] >= start_x) & \
                    (ring[:, 0] < end_x) & \
                    (ring[:, 1] >= start_y) & \
                    (ring[:, 1] < end_y)

            if within_bbox_mask.all():
                idxes.append(i)

        return idxes

    def get_within_bbox_mask(self, points, start_x, start_y, end_x, end_y):
        within_bbox_mask = \
                (points[:, 0] >= start_x) & \
                (points[:, 0] < end_x) & \
                (points[:, 1] >= start_y) & \
                (points[:, 1] < end_y)
        return within_bbox_mask

    def assign_points_to_boxes(self, points, np_boxes):
        boxes = torch.tensor(np_boxes, device=points.device)
        boxes_center = torch.stack([(boxes[:,0] + boxes[:,2]) / 2., (boxes[:,1] + boxes[:,3]) / 2.], dim=1)
        dis = points.view(-1, 1, 2) - boxes_center.view(1, -1, 2)
        assignment = torch.abs(dis).sum(dim=-1).argmin(dim=1)
        return assignment

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

    def sample_points_in_ring(self, ring, interval=None, add_noise=False):

        interval = self.ring_sample_conf['interval'] if interval is None else interval

        ring_shape = shapely.LinearRing(ring)
        perimeter = ring_shape.length
        # num_bins = round(perimeter / interval)
        num_bins = max(round(perimeter / interval), 4)
        # num_bins = max(round(perimeter / interval), 4)
        bins = np.linspace(0, 1, num_bins)
        sampled_points = [ring_shape.interpolate(x, normalized=True) for x in bins]
        sampled_points = [[temp.x, temp.y] for temp in sampled_points]

        if add_noise:
            noise_type = self.ring_sample_conf.get('noise_type', 'uniform')

            if noise_type == 'random':
                noise_type = random.choice(['uniform', 'skip'])

            if noise_type == 'uniform':
                noise = (np.random.rand(len(sampled_points), 2) - 0.5) * interval / 2.01
            elif noise_type == 'skip':
                noise = (np.random.rand(len(sampled_points), 2) - 0.5) * interval / 1.01
                noise[0:2:-1] = 0
            elif noise_type == 'none':
                noise = 0

            if len(sampled_points) > 0:
                sampled_points = (np.array(sampled_points) + noise).tolist()
                sampled_points.append(sampled_points[0])

        return sampled_points

    def rasterize(self, batch_features, downscale=4, raster_shape=(256, 256), add_noise=False,
                  filter_small=True, all_touched=True):

        rasters = []
        for features in batch_features:
            shapes = []
            cnt = 1
            for feat in features:
                new_rings = []
                for ring in feat['coordinates']:
                    # exterior = (np.array(feat['exterior']) / downscale - offset).tolist()
                    # interiors = [(np.array(x) / downscale - offset).tolist() for x in feat['interiors']]
                    norm_ring = (ring / downscale).tolist()

                    if add_noise:
                        noisy_ring = self.sample_points_in_ring(norm_ring, add_noise=add_noise)
                        new_rings.append(noisy_ring)
                    else:
                        new_rings.append(norm_ring)

                exterior = new_rings[0]
                interiors = [] if len(new_rings) == 1 else new_rings[1:]
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

    def forward_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)

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

    def normalize_coordinates(self, graph, ws, input):
        if input == 'global':
            graph = (graph * 2 / ws - 1)
        elif input == 'normalized':
            graph = ((graph + 1) * ws / 2)
            # graph = torch.round(graph).long()
            graph[graph < 0] = 0
            graph[graph >= ws] = ws - 1
        return graph

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


            # img[0, contours[:,1], contours[:,0]]

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
        # gt_features = [x['features'] for x in gt_features]
        img_metas = kwargs['img_metas']

        h_crop, w_crop = self.train_cfg.crop_size
        assert h_crop == w_crop
        H, W = self.train_cfg.get('raster_shape', (1024, 1024))

        # B, _, H, W = ori_img.size()
        device = (self.backbone.parameters()).__next__().device

        raster_input = self.rasterize(
            gt_features, add_noise=True, downscale=4,
            all_touched=self.train_cfg.get('all_touched', True)
        )
        raster_gt = self.rasterize(
            gt_features, raster_shape=(H, W), downscale=1,
            all_touched=self.train_cfg.get('all_touched', True)
        )

        if len(raster_gt) == 1:
            pdb.set_trace()

        raster_input = torch.tensor(raster_input, device=device).float()
        raster_gt = torch.tensor(raster_gt, device=device).float()

        img = F.interpolate(raster_input.unsqueeze(1), size=(H, W), mode='nearest').int()
        img = (img > 0).float() - 0.5
        img = img.repeat(1,3,1,1)
        x = self.extract_feat(img)

        # gt_semantic_seg = F.interpolate(raster_gt.unsqueeze(1), size=(H, W)).int()
        gt_semantic_seg = raster_gt.unsqueeze(1)
        gt_semantic_seg = (gt_semantic_seg > 0).long()

        if self.train_cfg.get('learn_res', False):
            gt_semantic_seg = (gt_semantic_seg - (img[:,0:1] > 0).long()) + 1

        losses = dict()
        states = dict()

        loss_decode, state = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg
        )
        # seg_logits = state['seg_logits']
        losses.update(loss_decode)
        states.update(state)
        states['noise_imgs'] = img
        states['raster_gt'] = gt_semantic_seg

        """
        x_ori = self.extract_feat(ori_img)
        seg_logits, state_decode = self.decode_head.forward(x_ori)
        states['seg_logits'] = seg_logits

        pred_res = seg_logits.max(dim=1)[1]
        pred_res = F.interpolate(pred_res.float().unsqueeze(1), size=(H, W))[:,0]

        pred_bf = (img.clone()[:,0].reshape(-1) > 0).float()
        pred_bf[pred_res.reshape(-1) == 0] = 0
        pred_bf[pred_res.reshape(-1) == 2] = 1
        states['pred_bf'] = (pred_bf - 0.5).view(B, 1, H, W).repeat(1,3,1,1)

        """

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
        states['gdal_polygons'] = gdal_rings
        states['gt_polygons'] = gt_rings

        return losses, states

    def get_crop_boxes(self, img_H, img_W, crop_size=256, stride=192):
        # prepare locations to crop
        num_rows = math.ceil((img_H - crop_size) / stride) if math.ceil(
            (img_W - crop_size) /
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

    def get_patch_weight(self, patch_size):
        choice = 1
        if choice == 0:
            step_size = (1.0 - 0.5)/(patch_size/2)
            a = np.arange(1.0, 0.5, -step_size)
            b = a[::-1]
            c = np.concatenate((b,a))
            ct = c.reshape(-1,1)
            x = ct*c
            return x
        elif choice == 1:
            min_weight = 0.5
            step_count = patch_size//4
            step_size = (1.0 - min_weight)/step_count
            a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
            a = a * min_weight
            for i in range(1, step_count + 1):
                a[i:-i, i:-i] += step_size
            a = cv2.GaussianBlur(a,(5,5),0)
            return a
        else:
            a = np.ones(shape=(patch_size,patch_size), dtype=np.float32)
            return a

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, contours=None, contour_labels=None,
                        comp_mask=None, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        device = (self.backbone.parameters()).__next__().device

        img = torch.stack(img, dim=0)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        ## TODO: support different stride size
        # assert (h_stride == h_crop), 'currently only support stride==crop_size'
        # assert (w_stride == w_crop), 'currently only support stride==crop_size'

        assert h_crop == w_crop
        assert h_stride == w_stride
        scale_factor = self.test_cfg['scale_factor']

        if scale_factor > 1:
            h_crop = h_crop // scale_factor
            w_crop = w_crop // scale_factor
            h_stride = h_stride // scale_factor
            w_stride = w_stride // scale_factor

        batch_size, _, h_img, w_img = img.size()
        crop_boxes = self.get_crop_boxes(h_img, w_img, h_crop, h_stride)

        out_scale_factor = scale_factor if self.test_cfg.get('upscale_output', True) else 1
        num_classes = 2
        preds = torch.zeros((batch_size, num_classes, h_img * out_scale_factor, w_img * out_scale_factor))
        count_mat = torch.zeros((batch_size, 1, h_img * out_scale_factor, w_img * out_scale_factor))
        weights = torch.tensor(self.get_patch_weight(h_crop))

        assert batch_size == 1
        split_batch_size = 8

        for i in range(batch_size):
            crop_imgs = []
            for crop_idx, crop_box in enumerate(crop_boxes):
                start_x, start_y, end_x, end_y = crop_box
                crop_img = img[i:i+1, :, start_y:end_y, start_x:end_x]
                crop_imgs.append(crop_img)

            start_idx = 0
            stop = len(crop_imgs) if len(crop_imgs) % split_batch_size != 0 else len(crop_imgs) + split_batch_size
            splits = list(np.arange(0, stop, split_batch_size))

            for j in tqdm(range(len(splits) - 1)):
                cur_img = torch.cat(crop_imgs[splits[j]:splits[j+1]])
                cur_img = F.interpolate(cur_img, scale_factor=(scale_factor, scale_factor), mode='nearest').to(device)

                x = self.extract_feat(cur_img)
                crop_seg_logit, cur_state = self._decode_head_forward_test(x, img_meta)

                for crop_idx, crop_box in enumerate(crop_boxes[splits[j]:splits[j+1]]):
                    start_x, start_y, end_x, end_y = crop_box

                    temp = crop_seg_logit[crop_idx:crop_idx+1].cpu() * weights.view(1,1, h_crop, h_crop)
                    preds[i:i+1, :, start_y * out_scale_factor:end_y * out_scale_factor,
                          start_x * out_scale_factor:end_x * out_scale_factor] += temp

                    count_mat[i:i+1, :, start_y * out_scale_factor:end_y * out_scale_factor,
                          start_x * out_scale_factor:end_x * out_scale_factor] += 1

        preds = preds / (count_mat + 1e-8)
        # probs = F.softmax(preds, dim=-1)

        return preds

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

    def process_patch(self, img, contours, window_feats, contour_labels, offset, scale_factor):

        states = dict()
        x = self.extract_feat(img)
        seg_logits, state_decode = self.decode_head.forward(x)
        states.update(state_decode)
        feats = state_decode['feats']

        # loss_decode, state_decode = self._decode_head_forward_train(x, img_metas, mask)
        # losses.update(loss_decode)
        # states.update(state_decode)

        _, _, H, W = img.shape
        assert H == W # This make things easier
        proposals = self.get_proposals(contours, H, W, num_min_proposals=-1, device=img.device)
        if len(proposals) == 0:
            return [[]], [torch.zeros(0, 2, dtype=torch.int)]

        norm_proposals = self.point_head.normalize_coordinates(proposals, W, 'global')
        point_feats = F.grid_sample(
            feats[0:1],
            norm_proposals.view(1, 1, -1, 2),
            align_corners=True
        ).squeeze(2)[0].permute(1,0) # (num_points, num_channels)

        pred_point_cls = self.point_head(
            point_feats=point_feats,
            mode='point_cls',
            return_loss=False
        ).unsqueeze(0)

        pred_degrees = self.point_head(
            point_feats=point_feats,
            mode='angle',
            return_loss=False
        ).unsqueeze(0)

        pred_feats = window_feats.unsqueeze(0)

        # point_preds_ori = self.matching_head.normalize_coordinates(point_preds, W, 'normalized')
        point_preds_ori = proposals.unsqueeze(0)
        bbox_preds = self.generate_bbox_from_point(point_preds_ori[0], width=self.nms_width)

        # probs = torch.rand(bbox_preds.shape[0], device=bbox_preds.device)
        probs = F.softmax(pred_point_cls, dim=-1)[0, :, 1]

        # probs = F.softmax(cls_scores, dim=-1)[0, :, 1]
        keep_idx = mmcv.ops.nms(bbox_preds, probs, iou_threshold=0.0)[1].cpu().numpy()
        prob_idx = torch.nonzero(probs > self.prob_thre).squeeze(1).cpu().numpy()
        keep_idx = np.intersect1d(keep_idx, prob_idx)
        keep_idx = keep_idx[:self.num_max_test_nodes]

        if len(keep_idx) > 0:
            # keep_idx = torch.randperm(len(probs), device=probs.device)[:self.num_max_test_nodes]
            pos_point_preds = point_preds_ori[0, keep_idx]
            pos_point_preds = pos_point_preds * 1.0 / scale_factor + offset
            # pos_point_preds = point_preds[0, keep_idx[:self.num_max_test_nodes]]
            # pos_point_preds = proposals[keep_idx[:self.num_max_test_nodes]]
            # pos_point_preds = self.matching_head.normalize_coordinates(pos_point_preds, W, 'normalized')
            # pos_point_feats = point_feats[0, keep_idx]
            pos_point_feats = pred_feats[0, keep_idx]
            # pos_contour_labels = contour_labels[keep_idx[:self.num_max_test_nodes]]
            # comp_mask = self.get_comp_mask(pos_contour_labels)

            cur_comp_idxes = contour_labels.unique()
            for idx in cur_comp_idxes:
                idx = idx.item()
                # self.cur_num_points_per_comp[idx] += (contour_labels == idx).sum()
                self.cur_num_patch_per_comp[idx] += 1
                comp_mask = (contour_labels[keep_idx] == idx)
                if not idx in self.point_feats_buffer.keys():
                    self.point_feats_buffer[idx] = pos_point_feats[comp_mask]
                    self.point_preds_buffer[idx] = pos_point_preds[comp_mask]
                elif self.point_feats_buffer[idx] is not None: # if it's none, it means it has been processed
                    self.point_feats_buffer[idx] = torch.cat([self.point_feats_buffer[idx],
                                                              pos_point_feats[comp_mask]], dim=0)
                    self.point_preds_buffer[idx] = torch.cat([self.point_preds_buffer[idx],
                                                              pos_point_preds[comp_mask]], dim=0)

                if self.cur_num_patch_per_comp[idx] == self.num_patch_per_comp[idx]:
                    self.dump_comp_feats(idx)
                    self.cur_num_patch_per_comp[idx] = 0
                elif self.cur_num_patch_per_comp[idx] > self.num_patch_per_comp[idx]:
                    pdb.set_trace()

            # polygons, vertices = self.matching_head(
            #     img, point_feats=pos_point_feats.unsqueeze(0),
            #     point_preds=pos_point_preds.unsqueeze(0),
            #     comp_mask=comp_mask.unsqueeze(0),
            #     return_loss=False
            # )
            return self.check_and_dump_polygons()
        else:
            return [[]], [torch.zeros(0, 2, dtype=torch.int)]

    def normalize_comp_points(self, points, buffer_ratio=0.2, max_w=256, max_h=256, method='standard'):
        max_x, max_y = points.max(dim=0)[0]
        min_x, min_y = points.min(dim=0)[0]

        W = (max_x - min_x)
        H = (max_y - min_y)

        if method == 'standard':
            a = max(W, H) * (1 + buffer_ratio)
            points[:, 0] -= min_x
            points[:, 1] -= min_y
            points += max(W, H) * (buffer_ratio / 2)
            new_points = self.point_head.normalize_coordinates(points, a, 'global')

        elif method == 'fix_scale':
            # if max_w >= W and max_h >= H:
            #     offset = torch.tensor([max_w - W, max_h - H], device=points.device)
            #     offset = random_offset * torch.rand(2, device=points.device)
            #     offset -= torch.tensor([min_x, min_y], device=points.device)
            #     new_points = points + random_offset
            #     new_points = self.matching_head.normalize_coordinates(new_points, max_w, 'global')
            # else:
            #     offset = torch.tensor([W, max_h - H], device=points.device)

            if max_w >= W:
                offset_x = torch.rand(1, device=points.device) * (max_w - W)
                offset_x -= min_x
            else:
                offset_x = - torch.ones(1, device=points.device) * (W - max_w) / 2.
                offset_x -= min_x

            if max_h >= H:
                offset_y = torch.rand(1, device=points.device) * (max_h - H)
                offset_y -= min_y
            else:
                offset_y = - torch.ones(1, device=points.device) * (H - max_h) / 2.
                offset_y -= min_y

            new_points = points + torch.cat([offset_x, offset_y]).view(1, 2)
            new_points = self.point_head.normalize_coordinates(new_points, max_w, 'global')


        return new_points

    def dump_comp_feats(self, comp_idx):
        point_feats = self.point_feats_buffer[comp_idx]
        self.point_preds_buffer[comp_idx] = self.point_preds_buffer[comp_idx]
        self.point_pool.append((point_feats, self.point_preds_buffer[comp_idx]))
        self.point_feats_buffer[comp_idx] = None
        self.point_preds_buffer[comp_idx] = None

    """
    def greedy_arange(self, sizes):

        sorted_idxes = np.argsort(np.array(sizes))[::-1]
        base_size = sizes[sorted_idxes[0]]
        batch_idx_list = [[sorted_idxes[0]]]
        batch_size_list = [[base_size]]
        cur_bin = []
        cur_sizes = []
        cur_size = 0
        for i, idx in enumerate(sorted_idxes[1:]):
            if sizes[idx] < 4:
                break
            if cur_size + sizes[idx] <= base_size:
                cur_bin.append(idx)
                cur_sizes.append(sizes[idx])
                cur_size += sizes[idx]
            else:
                batch_idx_list.append(cur_bin)
                batch_size_list.append(cur_sizes)
                cur_bin = [idx]
                cur_size = sizes[idx]
                cur_sizes = [sizes[idx]]

        if len(cur_bin) > 0:
            batch_idx_list.append(cur_bin)
            batch_size_list.append(cur_sizes)

        return batch_idx_list, batch_size_list
    """

    def greedy_arange(self, sizes, max_base_size):

        sorted_idxes = np.argsort(np.array(sizes))[::-1]
        start_idx = (sizes > max_base_size).sum()
        if (sizes > max_base_size).all():
            return [], []
        base_size = sizes[sorted_idxes[start_idx]]
        batch_idx_list = [[sorted_idxes[start_idx]]]
        batch_size_list = [[base_size]]
        cur_bin = []
        cur_sizes = []
        cur_size = 0
        for i, idx in enumerate(sorted_idxes[start_idx+1:]):
            if sizes[idx] < 4:
                break
            if sizes[idx] > max_base_size:
                continue
            if cur_size + sizes[idx] <= base_size:
                cur_bin.append(idx)
                cur_sizes.append(sizes[idx])
                cur_size += sizes[idx]
            else:
                batch_idx_list.append(cur_bin)
                batch_size_list.append(cur_sizes)
                cur_bin = [idx]
                cur_size = sizes[idx]
                cur_sizes = [sizes[idx]]

        if len(cur_bin) > 0:
            batch_idx_list.append(cur_bin)
            batch_size_list.append(cur_sizes)

        return batch_idx_list, batch_size_list



    def check_and_dump_polygons(self):
        sizes = np.array([len(feat) for feat, pred in self.point_pool])
        if sizes.sum() > 2048:
            device = self.point_pool[0][0].device
            batch_idx_list, batch_size_list = self.dqn_head.greedy_arange(sizes, self.dqn_head.max_base_size)
            # base_size = sizes[batch_idx_list[0][0]]
            sizes_with_terminals = [sum(sizes_list) + len(sizes_list) for sizes_list in batch_size_list]
            base_size = np.array(sizes_with_terminals).max()

            point_feats = torch.zeros(len(batch_idx_list),
                                      base_size,
                                      self.point_pool[0][0].shape[-1],
                                      device=device)
            point_preds = torch.zeros(len(batch_idx_list),
                                      base_size,
                                      self.point_pool[0][1].shape[-1],
                                      device=device)
            point_preds_ori = torch.zeros(len(batch_idx_list),
                                          base_size,
                                          self.point_pool[0][1].shape[-1],
                                          device=device)
            comp_masks = torch.zeros(len(batch_idx_list),
                                     base_size,
                                     base_size,
                                     device=device,
                                     dtype=torch.int)

            dummy_points = torch.zeros(1, 2, device=device) - 1
            dummy_feats = torch.zeros(1, self.point_pool[0][0].shape[-1], device=device) - 1
            for i, batch_idxes in enumerate(batch_idx_list):
                # batch_point_feats = [self.point_pool[idx][0] for idx in batch_idxes]
                batch_point_feats = [torch.cat([self.point_pool[idx][0], dummy_feats], dim=0) for idx in batch_idxes]
                # batch_point_preds_with_dummy = [torch.cat([self.point_pool[idx][1], dummy_points], dim=0) for idx in batch_idxes]
                batch_point_preds = [
                    torch.cat([
                        self.normalize_comp_points(
                            self.point_pool[idx][1], method=self.comp_norm_mode
                        ), dummy_points
                    ]) for idx in batch_idxes
                ]
                batch_point_preds_ori = [
                    torch.cat([
                        self.point_pool[idx][1],
                        dummy_points
                    ]) for idx in batch_idxes
                ]

                batch_point_feats = torch.cat(batch_point_feats, dim=0)
                batch_point_preds = torch.cat(batch_point_preds, dim=0)
                batch_point_preds_ori = torch.cat(batch_point_preds_ori, dim=0)

                point_feats[i, :len(batch_point_feats)] = batch_point_feats
                point_preds[i, :len(batch_point_preds)] = batch_point_preds
                point_preds_ori[i, :len(batch_point_preds_ori)] = batch_point_preds_ori

                cur_idx = 0
                for size in batch_size_list[i]:
                    temp = comp_masks[i, cur_idx:cur_idx+size]
                    temp[:, cur_idx:cur_idx+size] = 1
                    comp_masks[i, cur_idx:cur_idx+size] = temp
                    cur_idx += size

            games = dict(
                points=point_preds,
                points_ori=point_preds_ori,
                point_feats=point_feats,
                batch_sizes=batch_size_list,
                batch_idxes=batch_idx_list,
                # poly_idxes_list=self.poly_idxes_list,
                # batch_gt_polygons=batch_gt_polygons
            )
            results_target = self.dqn_head.evaluate(games, net_type='target')
            polygons = []
            for key, value in results_target['return_polygons'].items():
                cur_polygons = [torch.tensor(x) for x in value]
                polygons.append(cur_polygons)

            self.point_pool = []
            return polygons, [torch.zeros(0, 2, dtype=torch.int)]

            # self.dqn_head.set_environment(**dqn_states, cur_iter=0)
            # games = self.dqn_head.sample_games()

            polygons, vertices = self.dqn_head(
                point_feats=point_feats,
                point_preds=point_preds,
                comp_mask=comp_masks,
                point_preds_ori=point_preds_ori,
                batch_size_list=batch_size_list,
                return_loss=False,
                mode='global'
            )
            polygons, vertices = self.matching_head(
                point_feats=point_feats,
                point_preds=point_preds,
                comp_mask=comp_masks,
                point_preds_ori=point_preds_ori,
                return_loss=False,
                mode='global'
            )
            self.point_pool = []
            return polygons, vertices

        return [[]], [torch.zeros(0, 2, dtype=torch.int)]

    def get_comp_mask(self, contour_labels):
        num_points = len(contour_labels)
        graph_mask = torch.zeros(num_points, num_points, dtype=torch.int).to(contour_labels.device)

        for comp_idx in contour_labels.unique():
            if comp_idx == -1:
                continue
            temp = (contour_labels == comp_idx)
            temp2 = graph_mask[:num_points, :num_points][temp]
            temp2[:, temp] = 1
            graph_mask[:num_points, :num_points][temp] = temp2

        return graph_mask


    def generate_bbox_from_point(self, points, width=3):
        start = points - width / 2.
        end = points + width / 2.
        bboxes = torch.cat([start, end], dim=1)
        return bboxes



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
            preds = self.slide_inference(
                img, img_meta, rescale, contours=contours, contour_labels=contour_labels, **kwargs
            )
            result['seg_logits'] = preds

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


