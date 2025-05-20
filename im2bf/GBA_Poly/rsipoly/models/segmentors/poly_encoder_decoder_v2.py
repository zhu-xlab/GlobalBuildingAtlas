# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import numpy as np

from rsipoly.core import add_prefix
from rsipoly.ops import resize
from .. import builder
from ..builder import SEGMENTORS
# from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from mmcv.parallel import DataContainer
from pycocotools import mask as cocomask


@SEGMENTORS.register_module()
class PolyEncoderDecoderV2(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, backbone, decode_head, point_det_head, matching_head, suppression, wandb_cfg=None, **kwargs):
        super(PolyEncoderDecoderV2, self).__init__(backbone, decode_head, **kwargs)
        self.matching_head = builder.build_head(matching_head)
        self.suppression = builder.build_head(suppression)
        self.point_det_head = builder.build_head(point_det_head)
        self.wandb_cfg = wandb_cfg

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
        # H, W, C = data_batch['img_shape']
        losses, states = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        # states.update({'img': data_batch['img'], 'gt': data_batch['mask']})
        states.update({'img': data_batch['img']})
        # states['vis|masks_gt'] = [data_batch['img'], data_batch['mask'][:, 0].cpu().numpy()]

        # probs = F.softmax(states['seg_logits'], dim=1)[:, 1:, ...].detach()
        # states['vis|featmap_probs'] = [data_batch['img'], probs]
        states['vis|points_vertices'] = [data_batch['img'], data_batch['contours']]
        states['vis|points_pos_point_preds'] = [data_batch['img'], states['pos_point_preds']]
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states)

        return outputs

    def forward_train(self, img, img_metas, contours, gt_points, polygons, **kwargs):
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
        contours = [torch.tensor(contour).to(img.device) for contour in contours]
        gt_points = [torch.tensor(gt_point).to(img.device) for gt_point in gt_points]
        # gt_labels = [torch.ones(len(gt_point)).to(img.device).to(torch.int) for gt_point in gt_points]

        x = self.extract_feat(img)

        losses = dict()
        states = dict()

        seg_logits, state_decode = self.decode_head.forward(x)
        states.update(state_decode)
        feats = state_decode['feats']

        # loss_decode, state_decode = self._decode_head_forward_train(x, img_metas, mask)
        # losses.update(loss_decode)
        # states.update(state_decode)

        proposals = self.get_proposals(contours)

        # def forward(self, img, feats, proposals, gt_points, gt_labels):

        loss_det, state_det = self.point_det_head(
            img, feats=feats, proposals=contours,
            gt_points=gt_points, img_metas=img_metas
        )
        losses.update(loss_det)
        states.update(state_det)

        pos_point_preds = state_det['pos_point_preds']
        point_feats = state_det['point_feats']
        gt_inds = state_det['gt_inds']
        pos_inds = state_det['pos_inds']
        labels = state_det['labels']

        graph_targets = self.get_graph_targets(pos_inds, gt_inds, gt_points, polygons)

        # merged_vertices, merged_vertices_mask, merged_permute_matrix = \
        #         self.merge_preds_gts(pred_points, vertices, vertices_mask, permute_matrix)

        pos_point_feats = [point_feat[label > 0] for point_feat, label in zip(point_feats, labels)]
        loss_matching, state_matching = self.matching_head(
            img, point_feats=pos_point_feats, graph_targets=graph_targets, pos_point_preds=pos_point_preds,
            return_loss=True
        )
        losses.update(loss_matching)


        # if self.with_auxiliary_head:
        #     loss_aux, state_aux = self._auxiliary_head_forward_train(
        #         x, img_metas, mask)
        #     losses.update(loss_aux)
        #     states.update(state_aux)

        return losses, states

    def get_proposals(self, contours):
        return contours

    def get_graph_targets(self, pos_inds, gt_inds, gt_points, polygons_list):

        graph_targets = []
        for i in range(len(polygons_list)):
            polygons = polygons_list[i]
            pos_ind = pos_inds[i]
            pos2gt_ind = gt_inds[i][pos_inds[i]] - 1
            gt_point = gt_points[i]
            poly_sizes = torch.tensor([len(polygon) for polygon in polygons], dtype=torch.long).to(pos_ind.device)
            num_vertices = sum(poly_sizes)
            num_points = len(pos_ind)
            gt2pos_ind = torch.zeros((num_points,), device=pos_ind.device, dtype=torch.long)
            gt2pos_ind[pos2gt_ind] = torch.arange(0, num_points, device=pos_ind.device, dtype=torch.long)

            if num_vertices > 0:
                col_idx = list(range(1, num_points+1))

                offsets = []
                start_idx = 0
                for poly_size in poly_sizes:
                    if start_idx + poly_size.item() > num_points:
                        col_idx[-1] = 0
                        break
                    start_idx += poly_size.item()
                    offsets.append(start_idx-1)

                col_idx = torch.tensor(col_idx, dtype=torch.long).to(pos_ind.device)
                # offsets = torch.tensor(offsets, dtype=torch.long).to(pos_ind.device)
                col_idx[offsets] -= poly_sizes[:len(offsets)]
                row_idx = torch.arange(0, num_points, dtype=torch.long, device=pos_ind.device)

                assert len(pos2gt_ind) == num_points

                permute_mat = torch.zeros(num_points, num_points, dtype=torch.int).to(pos_ind.device)
                # permute_mat[row_idx[gt_ind], col_idx[gt_ind]] = 1
                # if num_vertices > num_points:
                assert (col_idx >= num_points).sum() == 0
                assert pos2gt_ind.max() < num_points
                # pdb.set_trace()
                # col_idx[col_idx >= num_points] = (col_idx >= num_points).nonzero() # Form a self loop for the polygons that exceed the maximum limit
                permute_mat[row_idx[pos2gt_ind], gt2pos_ind[col_idx]] = 1
            else:
                permute_mat = torch.zeros(num_vertices, num_vertices, dtype=torch.int).to(pos_ind.device)

            graph_targets.append(permute_mat)

        return graph_targets


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


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, corner_points=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        ## TODO: support different stride size
        assert (h_stride == h_crop), 'currently only support stride==crop_size'
        assert (w_stride == w_crop), 'currently only support stride==crop_size'

        assert h_crop == w_crop
        assert h_stride == w_stride
        batch_size, _, h_img, w_img = img.size()
        crop_boxes = self.get_crop_boxes(h_img, w_img, h_crop, h_stride)
        assert batch_size == 1
        # num_classes = self.num_classes
        # h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        # w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        # preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        # count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        polygons = []
        vertices = []

        for i in range(batch_size):

            cur_polygons = []
            cur_vertices = []
            for img_id, crop_box in enumerate(crop_boxes):
                start_x, start_y, end_x, end_y = crop_box
                crop_img = img[i:i+1, :, start_y:end_y, start_x:end_x]
                offset_torch = torch.tensor((start_x, start_y)).view(1, 2).to(crop_img.device)
                offset_np = np.array((start_x, start_y)).reshape(1, 2)

                cur_corner_points = None
                if corner_points is not None:
                    corner_points[i] = corner_points[i].view(-1, 2)
                    if len(corner_points[i]) > 0:
                        # cur_corner_points = [vertex - offset for vertex in corner_points if start_x <= vertex[0] < end_x and start_y <= vertex[1] < end_y]
                        within_bbox_mask = \
                                (corner_points[i][:, 0] >= start_x) & \
                                (corner_points[i][:, 0] < end_x) & \
                                (corner_points[i][:, 1] >= start_y) & \
                                (corner_points[i][:, 1] < end_y)
                        cur_corner_points = corner_points[i][within_bbox_mask] - offset_torch
                    else:
                        cur_corner_points = corner_points[i]

                seg_logit, cur_state = self.encode_decode(crop_img, [img_meta[i]])
                feats = cur_state['feats']

                crop_polygons, crop_vertices = self.get_polygons(crop_img, seg_logit, feats, [cur_corner_points])
                for poly_idx, temp in enumerate(crop_polygons[0]):
                    if len(temp) > 0:
                        crop_polygons[0][poly_idx] += offset_np

                cur_polygons.extend(crop_polygons[0])
                cur_vertices.extend(crop_vertices[0])

            # if len(cur_polygons) > 0:
            #     cur_polygons = np.concatenate(cur_polygons, axis=0)
            # else:
            #     cur_polygons = np.zeros((0, 2)).astype(np.int)

            if len(cur_vertices) > 0:
                cur_vertices = torch.cat(cur_vertices, dim=0)
            else:
                cur_vertices = torch.zeros(0, 2).to(img.device).to(torch.int)

            polygons.append(cur_polygons)
            vertices.append(cur_vertices)

        return polygons, vertices

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

    def get_polygons(self, img, seg_logit, feats, corner_points=None):

        probs = F.softmax(seg_logit, dim=1)
        B, _, H, W = probs.shape

        if corner_points is not None:
            corner_mask = torch.zeros(B, 1, H, W).to(torch.uint8).to(probs.device)
            for i in range(B):
                # corner_points[i] = corner_points[i].view(-1, 2)
                if corner_points[i].shape[0] > 0:
                    corner_mask[i, :, corner_points[i][:, 1], corner_points[i][:, 0]] = 1
            unfold_mask = F.unfold(corner_mask.float(), self.kernel_size, dilation=1, padding=self.kernel_size//2)
            unfold_mask = unfold_mask.view(B, -1, H, W).sum(dim=1) > 0
            probs = probs * unfold_mask

        _, vertices = self.suppression(probs[:, 1:, :, :]) # input the probability of building class
        # if len(vertices[0]) > 0:
        #     pdb.set_trace()
        polygons = self.matching_head(img, feats=feats, vertices_list=vertices, return_loss=False)
        return polygons, vertices

    def inference(self, img, img_meta, rescale, corner_points=None):
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
            polygons, vertices = self.slide_inference(img, img_meta, rescale, corner_points=corner_points)
            result['vis|polygons'] = [img, polygons]
            result['vis|points_corners'] = [img, corner_points]
            result['polygons'] = polygons
        else:
            seg_logit, state = self.whole_inference(img, img_meta, rescale)
            feats = state['feats']
            corner_points = [corner_point.view(-1, 2) for corner_point in corner_points]
            polygons, vertices = self.get_polygons(img, seg_logit, feats, corner_points)
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
            result['vis|points_corners'] = [img, corner_points]
            result['vis|points_vertices'] = [img, vertices]

        return result


    def simple_test(self, img, img_meta, corner_points=None, rescale=True):
        """Simple test with single image."""
        results = self.inference(img, img_meta, rescale, corner_points=corner_points)
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
