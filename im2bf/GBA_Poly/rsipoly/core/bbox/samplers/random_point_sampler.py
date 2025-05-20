# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import pdb
from rsidet.core.bbox.builder import BBOX_SAMPLERS

import torch

@BBOX_SAMPLERS.register_module()
class RandomPointSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=False,
                 num_max_gt=512,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self
        self.num_max_gt = 512

    def sample(self,
               assign_result,
               points, # (N, 2)
               gt_points,
               # gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.
        """

        gt_flags = points.new_zeros((points.shape[0], ), dtype=torch.uint8)

        if self.add_gt_as_proposals and len(gt_points) > 0:
            points = torch.cat([gt_points, points], dim=0)
            gt_ones = points.new_ones(gt_points.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = min(int(self.num * self.pos_fraction), self.num)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos)

        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg)
        neg_inds = neg_inds.unique()
        num_sampled_neg = neg_inds.numel()

        gt_labels = points.new_zeros((num_sampled_pos + num_sampled_neg,), dtype=torch.uint8)
        gt_labels[:num_sampled_pos] = 1

        # if num_sampled_pos + num_sampled_neg < 512:
        #     pdb.set_trace()

        pos_gt_inds = assign_result.gt_inds[pos_inds]


        sampling_result = dict(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            pos_gt_points=gt_points[pos_gt_inds-1],
            pos_points=points[pos_inds],
            neg_points=points[neg_inds],
            points=torch.cat([points[pos_inds], points[neg_inds]], dim=0),
            assign_result=assign_result,
            gt_flags=gt_flags,
            gt_inds = assign_result.gt_inds
        )

        return sampling_result

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/rsidetection/pull/5014
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        temp = assign_result.gt_inds > 0
        if self.num_max_gt > 0:
            temp = temp & (assign_result.gt_inds <= self.num_max_gt)

        pos_inds = torch.nonzero(temp, as_tuple=False)
        # if self.num_max_gt > 0:
        #     pos_inds = pos_inds[:self.num_max_gt]

        # if pos_inds.numel() != 0:
        pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        # if neg_inds.numel() != 0:
        neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
