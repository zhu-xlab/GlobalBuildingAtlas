import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from mmcv.runner import BaseModule
import torch.nn.functional as F
import pdb
from rsidet.models.utils import build_linear_layer
from rsidet.core import bbox2result, bbox2roi, build_assigner, build_sampler

from ..builder import HEADS, build_head

@HEADS.register_module()
class PointDetHead(BaseModule):
    def __init__(self,
                 point_head=None,
                 add_logits_offset=False,
                 use_logits_offset=False,
                 update_point_feats=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal', std=0.01
                 ),
                 **kwargs):
        super(PointDetHead, self).__init__(init_cfg)
        self.point_head = build_head(point_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_assigner_sampler()
        self.add_logits_offset = add_logits_offset
        self.use_logits_offset = use_logits_offset
        self.update_point_feats = update_point_feats

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.assigner = None
        self.sampler = None
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, img, img_metas, feats, proposals, gt_points, seg_logits=None):
        B, _, H, W = img.shape
        assert H == W

        sampling_results = []
        for i in range(B):
            proposals[i] = proposals[i].float()
            gt_points[i] = gt_points[i].float()

            assign_result = self.assigner.assign(
                proposals[i], gt_points[i], img_metas[i]
            )

            # pos_inds = torch.nonzero(
            #     assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
            # neg_inds = torch.nonzero(
            #     assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
            # gt_flags = proposals[i].new_zeros(proposals[i].shape[0], dtype=torch.uint8)

            # sampling_result = SamplingResult(pos_inds, neg_inds, proposals[i], gt_points[i],
            #                                  assign_result, gt_flags)
            sampling_result = self.sampler.sample(
                assign_result,
                proposals[i],
                gt_points[i],
            )
            sampling_results.append(sampling_result)

        point_targets, point_weights, labels = self.point_head.get_targets(sampling_results, img_H=H)
        points = torch.stack([sampling_result['points'] for sampling_result in sampling_results], dim=0)
        point_targets = torch.stack(point_targets, dim=0)
        point_weights = torch.stack(point_weights, dim=0)
        labels = torch.stack(labels, dim=0)

        _, N = labels.shape

        gt_inds_list = [res['gt_inds'] for res in sampling_results]
        pos_points_list = [res['pos_points'] for res in sampling_results]
        # pos_gt_points_list = [res['pos_gt_points'] for res in sampling_results]

        cls_scores, point_preds, point_feats = self.point_head(img, points, feats)
        if self.add_logits_offset:
            # grid = self.normalize_coordinates(point_p, W, input="global")
            point_logits = F.grid_sample(seg_logits, point_preds.unsqueeze(1), align_corners=True)
            point_logits = point_logits.squeeze(2).permute(0, 2, 1) # (1, N, C)
            cls_scores += point_logits

        elif self.use_logits_offset:
            point_logits = F.grid_sample(seg_logits, point_preds.unsqueeze(1), align_corners=True)
            point_logits = point_logits.squeeze(2).permute(0, 2, 1) # (1, N, C)
            cls_scores = point_logits

        if self.update_point_feats:
            point_feats = F.grid_sample(feats, point_preds.unsqueeze(1), align_corners=True)
            point_feats = point_feats.squeeze(2).permute(0, 2, 1) # (1, N, C)
            if self.point_head.add_pos_info:
                point_feats = torch.cat([point_feats, point_preds], dim=-1)

        losses = self.point_head.loss(cls_scores.reshape(B*N, -1),
                                      point_preds.view(B*N, 2),
                                      labels.view(B*N),
                                      point_targets.view(B*N, 2),
                                      label_weights=None,
                                      point_weights=point_weights.view(B*N, -1))
        states = dict(
            pos_inds=[res['pos_inds'] for res in sampling_results],
            gt_inds=gt_inds_list,
            pos_point_preds=pos_points_list,
            point_preds=point_preds,
            point_feats=point_feats,
            cls_scores=cls_scores,
            labels=labels,
        )

        # losses = dict()
        # bbox_results = self._bbox_forward_train(x, sampling_results,
        #                                         gt_points, gt_labels,
        #                                         img_metas)
        # losses.update(bbox_results['loss_bbox'])

        return losses, states

    def forward_test(self, img, feats, proposals, img_metas=None, seg_logits=None):
        cls_scores, point_preds, point_feats = self.point_head(img, proposals, feats)
        if self.add_logits_offset:
            # grid = self.normalize_coordinates(point_p, W, input="global")
            point_logits = F.grid_sample(seg_logits, point_preds.unsqueeze(1), align_corners=True)
            point_logits = point_logits.squeeze(2).permute(0, 2, 1) # (1, N, C)
            cls_scores += point_logits

        elif self.use_logits_offset:
            point_logits = F.grid_sample(seg_logits, point_preds.unsqueeze(1), align_corners=True)
            point_logits = point_logits.squeeze(2).permute(0, 2, 1) # (1, N, C)
            cls_scores = point_logits

        if self.update_point_feats:
            point_feats = F.grid_sample(feats, point_preds.unsqueeze(1), align_corners=True)
            point_feats = point_feats.squeeze(2).permute(0, 2, 1) # (1, N, C)
            if self.point_head.add_pos_info:
                point_feats = torch.cat([point_feats, point_preds], dim=-1)

        return cls_scores, point_preds, point_feats


    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.point_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.point_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.point_head.loss(bbox_results['cls_score'],
                                         bbox_results['bbox_pred'], rois,
                                         *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
