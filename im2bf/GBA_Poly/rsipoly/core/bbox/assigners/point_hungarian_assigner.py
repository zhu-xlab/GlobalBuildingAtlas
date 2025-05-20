# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb

from rsidet.core.bbox.builder import BBOX_ASSIGNERS
# from ..transforms import bbox_cxcywh_to_xyxy
from rsidet.core.bbox.assigners.assign_result import AssignResult
from rsidet.core.bbox.assigners.base_assigner import BaseAssigner
from rsidet.core.bbox.match_costs import build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

class PointL1Cost:

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, point_pred, gt_points):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(point_pred, gt_points, p=1)
        return bbox_cost * self.weight



@BBOX_ASSIGNERS.register_module()
class PointHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 add_gt_noises=False):

        # self.cls_cost = build_match_cost(cls_cost)
        # self.reg_cost = build_match_cost(reg_cost)
        self.reg_cost = PointL1Cost()
        self.add_gt_noises = add_gt_noises

    def assign(self,
               point_pred,
               gt_points_ori,
               img_meta=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_points = gt_points_ori.size(0), point_pred.size(0)

        if num_gts > num_points:
            gt_points = gt_points_ori[:num_points]
            num_gts = num_points
        else:
            gt_points = gt_points_ori

        # 1. assign -1 by default
        assigned_gt_inds = point_pred.new_full((num_points, ),
                                               -1,
                                               dtype=torch.long)
        if num_gts == 0 or num_points == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=None)

        # img_h, img_w, _ = img_meta['img_shape']
        # factor = gt_points.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # regression L1 cost
        # normalize_gt_points = gt_points / factor
        if self.add_gt_noises:
            noises = torch.randn(num_gts, 2, device=gt_points.device) * 1e-4
            reg_cost = self.reg_cost(point_pred, gt_points + noises)
        else:
            reg_cost = self.reg_cost(point_pred, gt_points)
        # regression iou cost, defaultly giou is used in official DETR.
        # bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        # iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs
        cost = reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            point_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            point_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=None)
