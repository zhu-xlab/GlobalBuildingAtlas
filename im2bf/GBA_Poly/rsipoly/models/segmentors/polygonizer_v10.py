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
from rsipoly.models.utils import PolygonProcessor
import rsipoly.utils.tanmlh_polygon_utils as polygon_utils
import geopandas as gpd

from rsipoly.ops import resize
from .. import builder
from ..builder import SEGMENTORS
# from .base import BaseSegmentor
from mmcv.runner import BaseModule
# from .encoder_decoder import EncoderDecoder
from mmcv.parallel import DataContainer
from pycocotools import mask as cocomask
from mmcv.runner import BaseModule
import torch.distributed as dist
from collections import OrderedDict
from rasterio.features import shapes
import shapely
import geojson
import rasterio
import networkx
from pathlib import Path
import os


@SEGMENTORS.register_module()
class PolygonizerV10(BaseModule):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self, regu_net, ring_sample_conf, train_cfg=None, test_cfg=None,
        init_cfg=None, **kwargs,
    ):
        super(PolygonizerV10, self).__init__(init_cfg)
        self.regu_net = builder.build_segmentor(regu_net)
        self.test_cfg = test_cfg
        self.init_weights()
        self.polygon_processor = PolygonProcessor(ring_sample_conf)

    def init_weights(self):
        # cache_file = hf_hub_download(repo_id=self.cfg['ckpt_repo_id'], filename=self.cfg['ckpt_filename'])

        regu_checkpoint = torch.load(self.init_cfg['regu_checkpoint'], map_location='cpu')
        log1 = self.regu_net.load_state_dict(
            regu_checkpoint['state_dict'],
            strict=False
        )


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

        # optimizer['generator'].zero_grad()
        losses, states = self(**data_batch)
        seg_logits = states['seg_logits']
        raster_gt = states['raster_gt']

        loss, log_vars = self._parse_losses(losses)
        # loss.backward()
        # optimizer['generator'].step()

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
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states
        )

        return outputs

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

        return losses, states

    # TODO refactor
    def slide_inference(self, imgs, img_meta, rescale, contours=None, contour_labels=None,
                        comp_mask=None, **kwargs):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        filename = str(img_meta[0]['filename'])
        in_root = str(img_meta[0]['in_root'])
        out_root = str(img_meta[0]['out_root'])
        geo_transform = img_meta[0]['geo_transform']

        if geo_transform is None:
            return []

        rel_path = os.path.relpath(filename, in_root)
        out_path = os.path.join(out_root, rel_path)
        out_path = out_path.split('.tif')[0]
        if 'crop_boxes' in img_meta[0]:
            crop_boxes = [str(x) for x in img_meta[0]['crop_boxes']]
            out_path = os.path.join(out_path, '_'.join(crop_boxes))

        print(out_path)

        """
        if not os.path.exists(out_path):
            try:
                os.makedirs(out_path)
            except Exception as e:
                print(e)
                return []
        else:
            return []

        """

        img = imgs[0]
        if (img == 1).sum() == 0:
            with open(os.path.join(out_path, 'finished.txt'), 'w') as out_file:
                pass
            return []

        nodata_mask = img == 255
        norm_img = torch.where(img == 1, 0.5, -0.5)

        device = (self.regu_net.parameters()).__next__().device
        preds = self.regu_net.slide_inference([norm_img], img_meta)
        preds = F.interpolate(preds, scale_factor=(2,2))

        probs = F.softmax(preds, dim=1)
        pred_mask = ((probs[:,1] > 0.5).float() - 0.5)
        pred_mask = pred_mask[0].unsqueeze(0)

        if self.test_cfg.get('save_regu_results', False):
            cur_pred_mask = pred_mask.cpu().numpy()[0]
            out_path = os.path.join(out_path, 'regu.tif')
            with rasterio.open(
                out_path, 'w', driver='GTiff',
                height=cur_pred_mask.shape[0],
                width=cur_pred_mask.shape[1],
                count=1,
                dtype=str(cur_pred_mask.dtype),
                crs=img_meta[0]['geo_crs'],
                transform=img_meta[0]['geo_transform']
            ) as dst:
                dst.write(cur_pred_mask, 1)

        # pred_polygons = self.polygon_processor.post_process_dp((pred_mask > 0).astype(torch.uint8))
        # pred_polygons = polygon_utils.((pred_mask > 0).astype(torch.uint8))

        pred_mask = pred_mask.cpu().numpy()[0]
        poly_jsons = list(rasterio.features.shapes((pred_mask > 0).astype(np.uint8), mask=pred_mask > 0))
        poly_jsons = [x[0] for x in poly_jsons]
        pred_polygons = polygon_utils.simplify_poly_jsons(
            poly_jsons, lam=5, max_step_size=128, interval=1, num_min_bins=16, format='json',
            device=img.device, verbose=True
        )
        self.format_results(pred_polygons, img_meta[0], upscale=1/2.)

        # pred_polygons = self.simp_net.slide_inference(pred_img, img_meta, rescale=rescale)

        # if len(pred_polygons) == 0:
        #     out_path = os.path.join(out_root, rel_path)
        #     pdb.set_trace()
        #     with open(os.path.join(out_path, 'finished.txt'), 'w') as out_file:
        #         pass

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
            polygons = self.slide_inference(
                img, img_meta, rescale, contours=contours, contour_labels=contour_labels, **kwargs
            )
            # result['polygons_v2'] = polygons

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
        return results

    def format_results(self, polygons, img_meta=None, upscale=1):

        if len(polygons) == 0:
            return None

        """Place holder to format result to dataset specific output."""
        filename = str(img_meta['filename'])
        city_transform = img_meta['geo_transform']
        crs = img_meta['geo_crs']

        in_root = str(img_meta['in_root'])
        out_root = str(img_meta['out_root'])

        rel_path = os.path.relpath(filename, in_root)
        out_path = os.path.join(out_root, rel_path)
        out_path = out_path.split('.tif')[0]

        if 'crop_boxes' in img_meta:
            crop_boxes = [str(x) for x in img_meta['crop_boxes']]
            out_path = os.path.join(out_path, '_'.join(crop_boxes))

        out_path += '.geojson'

        offset = np.array([0,0]).reshape(1,2)

        global_polygons = []

        for polygon in polygons:
            new_rings = []
            for ring in polygon['coordinates']:
                ring = torch.tensor(ring)
                new_ring = np.stack((city_transform * (ring * upscale + offset).permute(1,0).numpy()), axis=1)
                if len(new_ring) >= 4:
                    new_rings.append(new_ring)

            if len(new_rings) > 0:
                new_polygon = shapely.geometry.Polygon(new_rings[0], new_rings[1:] if len(new_rings) > 1 else None)
                global_polygons.append(new_polygon)

        gdf = gpd.GeoDataFrame(geometry=global_polygons)
        gdf.crs = crs
        gdf.to_file(out_path, driver='GeoJSON')
