# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
import pdb

from rsidet.core import build_bbox_coder, multi_apply, multiclass_nms
from rsidet.models.builder import build_loss
from rsidet.models.losses import accuracy
from rsidet.models.utils import build_linear_layer
from ..builder import HEADS

@HEADS.register_module()
class PointRegHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 in_channels=256,
                 num_classes=1,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 pos_enc_dim=-1,
                 add_pos_info=False,
                 use_point_offset=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_point=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg=None,
                 pos_weight=1.0):
        super(PointRegHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pos_weight = pos_weight
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.pos_enc_dim = pos_enc_dim
        self.add_pos_info = add_pos_info
        self.use_point_offset = use_point_offset


        self.loss_cls = build_loss(loss_cls)
        self.loss_point = build_loss(loss_point)

        in_channels = self.in_channels
        cls_channels = num_classes + 1
        self.fc_cls = build_linear_layer(
            self.cls_predictor_cfg,
            # in_features=in_channels + 0 if pos_enc_dim < 0 else pos_enc_dim,
            in_features=in_channels + (2 if add_pos_info else 0),
            out_features=cls_channels)

        out_dim_reg = 2 * num_classes
        self.fc_reg = build_linear_layer(
            self.reg_predictor_cfg,
            # in_features=in_channels + 0 if pos_enc_dim < 0 else pos_enc_dim,
            # in_features=in_channels,
            in_features=in_channels + (2 if add_pos_info else 0),
            out_features=out_dim_reg)

        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            self.init_cfg += [
                dict(
                    type='Normal', std=0.01, override=dict(name='fc_cls'))
            ]
            self.init_cfg += [
                dict(
                    type='Normal', std=0.001, override=dict(name='fc_reg'))
            ]

    def normalize_coordinates(self, graph, ws, input):
        if input == 'global':
            graph = (graph * 2 / ws - 1)
        elif input == 'normalized':
            graph = ((graph + 1) * ws / 2)
            graph = torch.round(graph).long()
            graph[graph < 0] = 0
            graph[graph >= ws] = ws - 1
        return graph

    @auto_fp16()
    def forward(self, img, points, feats, probs=None):
        # cls_score = self.fc_cls(x) if self.with_cls else None
        # point_pred = self.fc_reg(x) if self.with_reg else None
        # return cls_score, point_pred
        B, _, H, W = img.shape
        assert H == W

        if type(points) == list:
            point_preds = []
            cls_scores = []
            point_feats = []
            for i in range(B):
                if len(points[i]) > 0:
                    grid = self.normalize_coordinates(points[i].unsqueeze(0), W, input="global")
                    point_feat = F.grid_sample(feats[i:i+1], grid.unsqueeze(1), align_corners=True)
                    point_feat = point_feat.squeeze(2).permute(0, 2, 1) # (1, N, C)
                    point_pred = self.fc_reg(point_feat) # (1, N, 2)
                    cls_score = self.fc_cls(point_feat) # (1, N, 2)

                    point_preds.append(point_pred.squeeze(0))
                    cls_scores.append(cls_score.squeeze(0))
                    point_feats.append(point_feat.squeeze(0))

                else:
                    point_preds.append(torch.zeros(0, 2).to(img.device))
                    cls_scores.append(torch.zeros(0, self.num_classes+1).to(img.device))
                    point_feats.append(torch.zeros(0, self.in_channels).to(img.device))
        else:
            grid = self.normalize_coordinates(points, W, input="global")
            point_feats = F.grid_sample(feats, grid.unsqueeze(1), align_corners=True)
            point_feats = point_feats.squeeze(2).permute(0, 2, 1) # (1, N, C)
            if self.add_pos_info:
                point_feats = torch.cat([point_feats, grid], dim=-1)

            point_preds = self.fc_reg(point_feats) # (1, N, 2)
            cls_scores = self.fc_cls(point_feats) # (1, N, 2)

            if self.use_point_offset:
                point_preds = grid + point_preds



        # cls_scores = torch.cat(cls_scores, dim=0)
        # point_preds = torch.cat(point_preds, dim=0)

        return cls_scores, point_preds, point_feats

    def _get_target_single(self, pos_points, neg_points, pos_gt_points, img_H):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.
        """
        num_pos = pos_points.size(0)
        num_neg = neg_points.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        point_targets = pos_points.new_zeros(num_samples, 2)
        point_weights = pos_points.new_zeros(num_samples, 2)
        gt_labels = pos_points.new_zeros((num_samples,), dtype=torch.uint8)
        if num_pos > 0:
            pos_weight = 1.0 if self.pos_weight <= 0 else self.pos_weight
            pos_point_targets = self.normalize_coordinates(pos_gt_points.float(), img_H, input="global")

            point_targets[:num_pos, :] = pos_point_targets
            point_weights[:num_pos, :] = 1
            gt_labels[:num_pos] = 1

        return point_targets, point_weights, gt_labels

    def get_targets(self,
                    sampling_results,
                    img_H,
                    rcnn_train_cfg=None,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.
        """
        pos_points_list = [res['pos_points'] for res in sampling_results]
        neg_points_list = [res['neg_points'] for res in sampling_results]
        pos_gt_points_list = [res['pos_gt_points'] for res in sampling_results]

        # pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        point_targets, point_weights, gt_labels = multi_apply(
            self._get_target_single,
            pos_points_list,
            neg_points_list,
            pos_gt_points_list,
            img_H=img_H
        )
            # pos_gt_bboxes_list,
            # cfg=rcnn_train_cfg)

        return point_targets, point_weights, gt_labels

    @force_fp32(apply_to=('cls_scores', 'point_preds'))
    def loss(self,
             cls_scores,
             point_preds,
             labels,
             point_targets,
             label_weights=None,
             point_weights=None,
             reduction_override=None):
        losses = dict()
        if type(cls_scores) == list:
            cls_scores = torch.cat(cls_scores, dim=0)
            point_preds = torch.cat(point_preds, dim=0)
            labels = torch.cat(labels, dim=0)
            point_targets = torch.cat(point_targets, dim=0)
            if label_weights:
                label_weights = torch.cat(label_weights, dim=0)
            if point_weights:
                point_weights = torch.cat(point_weights, dim=0)

        if cls_scores is not None:
            # avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # avg_factor = None
            if cls_scores.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=None,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                    losses['acc'] = accuracy(cls_scores, labels)

        if point_preds is not None:
            # bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            pos_inds = labels > 0

            if pos_inds.any():

                # pos_point_pred = point_preds.view(
                #     point_preds.size(0), -1, 2)[pos_inds.type(torch.bool),
                #        labels[pos_inds.type(torch.bool)]]
                pos_point_pred = point_preds[pos_inds]

                losses['loss_point'] = self.loss_point(
                    pos_point_pred,
                    point_targets[pos_inds],
                    point_weights[pos_inds],
                    # avg_factor=point_targets.size(0),
                    avg_factor=None,
                    reduction_override=reduction_override)
            else:
                losses['loss_point'] = point_preds[pos_inds].sum()

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from rsidet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): Rois from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4) or
                (num_proposals, 5).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """

        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

