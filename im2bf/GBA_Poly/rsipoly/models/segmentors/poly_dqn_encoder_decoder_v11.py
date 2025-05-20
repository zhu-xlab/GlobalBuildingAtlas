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

from rsipoly.core import add_prefix
from rsipoly.ops import resize
from .. import builder
from ..builder import SEGMENTORS
# from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from mmcv.parallel import DataContainer
from pycocotools import mask as cocomask
from rsidet.core import bbox2result, bbox2roi, build_assigner, build_sampler


@SEGMENTORS.register_module()
class PolyDQNEncoderDecoderV11(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, backbone, decode_head, assigner, sampler, dqn_head, point_head, wandb_cfg=None,
                 comp_norm_mode='standard', use_point_feats=False, filter_separated_comp=False,
                 contour_noise_scale=0, num_dqn_update_per_step=100, freeze_base_net=False,
                 dqn_skill_level=4, matching_head=None, evaluate_freq=50, window_feat_size=6, **kwargs):
        super(PolyDQNEncoderDecoderV11, self).__init__(backbone, decode_head, **kwargs)
        self.dqn_head = builder.build_head(dqn_head)
        self.point_head = builder.build_head(point_head)
        if matching_head is not None:
            self.matching_head = builder.build_head(matching_head)
        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler)
        self.wandb_cfg = wandb_cfg
        self.num_max_test_nodes = 1024
        self.prob_thre = 0.0
        self.mask_graph = False
        self.nms_width = 7
        self.num_min_proposals = 512
        self.comp_norm_mode = comp_norm_mode
        self.use_point_feats = use_point_feats
        self.filter_separated_comp = filter_separated_comp
        self.contour_noise_scale = contour_noise_scale
        self.cur_iter = 0
        self.num_dqn_update_per_step = num_dqn_update_per_step
        self.freeze_base_net = freeze_base_net
        self.dqn_skill_level = dqn_skill_level
        self.num_game_success = {}
        base_net_list = [self.backbone, self.decode_head, self.point_head]
        if matching_head is not None:
            base_net_list.append(self.matching_head)
        self.base_net = torch.nn.ModuleList(base_net_list)
        self.act_net = self.dqn_head.act_net
        self.evaluate_freq = evaluate_freq
        self.window_feat_size = window_feat_size

        if self.test_cfg:
            self.num_max_test_nodes = self.test_cfg.get('num_max_test_nodes', 1024)
            self.prob_thre = self.test_cfg.get('prob_thre', 0.0)
            self.nms_width = self.test_cfg.get('nms_width', 7)
            self.test_scale_factor = self.test_cfg.get('scale_factor', 4)

        if self.train_cfg:
            self.mask_graph = self.train_cfg.get('mask_graph', False)
            self.num_min_proposals = self.train_cfg.get('num_min_proposals', 512)

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
        B, C, H, W = data_batch['img'].shape
        self.cur_iter += 1

        dqn_states = self(**data_batch, for_dqn=True)
        if dqn_states == {}:
            games = None
        else:
            self.dqn_head.set_environment(**dqn_states, cur_iter=0)
            games = self.dqn_head.sample_games()

        states = {}
        log_vars = {'success_iter': -1}

        if games is not None:

            success_vector = None
            success_rate = 0.
            for i in range(10):
                optimizer['act_net'].zero_grad()
                losses_dqn, states = self.dqn_head(games=games, return_loss=True)

                if losses_dqn == {}:
                    states = {}
                    log_vars = {'success_iter': -1}
                    break

                loss, log_vars = self._parse_losses(losses_dqn)
                loss.backward()
                optimizer['act_net'].step()
                num_games = len(states['pass_exam'])
                num_success = sum(states['pass_exam'])
                if success_vector is None:
                    success_vector = np.array(states['pass_exam'])
                else:
                    success_vector = success_vector | np.array(states['pass_exam'])

                success_rate = success_vector.mean()

                if success_vector.mean() >= 0.8:
                    break

            if self.evaluate_freq > 0 and self.cur_iter % self.evaluate_freq == 0:
                # eval_games = self.dqn_head.sample_games(add_dummy=False)
                results_act = self.dqn_head.evaluate(games, net_type='act')
                results_target = self.dqn_head.evaluate(games, net_type='target')
                success_rate_act = results_act['success_rate']
                success_rate_target = results_target['success_rate']
                success_cnt_act = results_act['success_cnt']
                success_cnt_target = results_target['success_cnt']
                all_cnt_act = results_act['all_cnt']
                all_cnt_target = results_target['all_cnt']
                success_iou_act = results_act['success_iou']
                success_iou_target = results_target['success_iou']

                s1 = set(success_rate_act.keys())
                s2 = set(success_rate_target.keys())
                scores = []
                levels = np.array(list(s1.union(s2)))
                levels.sort()
                for level in levels:
                    rate1 = success_rate_act.get(level, 0)
                    rate2 = success_rate_target.get(level, 0)
                    cnt1 = success_cnt_act.get(level, 0)
                    cnt2 = success_cnt_target.get(level, 0)
                    iou1 = success_iou_act.get(level, 0)
                    iou2 = success_iou_target.get(level, 0)
                    all_cnt1 = all_cnt_act.get(level, 0)
                    all_cnt2 = all_cnt_target.get(level, 0)

                    if abs(rate1 - rate2) < 1e-8:
                        scores.append(0)
                    elif rate1 > rate2:
                        scores.append(1)
                    else:
                        scores.append(-1)

                    print(f'level: {level}, act_net: {all_cnt1}, {cnt1}, {rate1:.3f}, {iou1:.3f},     ' +
                          f'target_net: {all_cnt2}, {cnt2}, {rate2:.3f}, {iou2:.3f}')

                if (np.array(scores) == 1).sum() > (np.array(scores) == -1).sum():
                    self.dqn_head.update_target_net()

                pred_polygons = []
                for key, value in results_target['return_polygons'].items():
                    cur_polygons = [torch.tensor(x) for x in value]
                    if len(cur_polygons) > 0:
                        pred_polygons += cur_polygons

                states['return_polygons'] = [pred_polygons]
                states['gt_polygons'] = [results_target['gt_polygons']]

                """
                else:
                    self.dqn_head.update_act_net()
                """
                # if success_rate_act >= 0.8:
                #     if game_level + 1 > self.dqn_skill_level:
                #         print(f'dqn skill level up to: {game_level+1}!')
                #         self.dqn_skill_level = game_level + 1


                """
                print(
                    # f"batch_size: {states['batch_size']}, node_num: {game_level}, iter: {states['iter']}, invalid0: {states['invalid0']}, invalid1: " +
                    # f"{states['invalid1']}, invalid2: {states['invalid2']}, valid: {states['valid']}, valid2: {states['valid2']}, " +
                    # f"iou_scores: {states['iou_scores']}, "
                    f"success_rate_act: {success_rate_act}, success_rate_target: {success_rate_target}"
                )
                """



        states['vis|masks_gt'] = [data_batch['img'], data_batch['mask'][:, 0].cpu().numpy()]

        if 'seg_logits' in states:
            probs = F.softmax(states['seg_logits'], dim=1)[:, 1:, ...].detach()
            states['vis|featmap_probs'] = [data_batch['img'], probs]
            states['vis|points_vertices'] = [data_batch['img'], data_batch['contours']]
            # states['vis|points_pos_point_preds'] = [data_batch['img'], states['pos_point_preds']]

        if 'point_preds_ori' in states:
            point_preds = states['point_preds_ori'].detach()
            states['vis|points_point_preds'] = [data_batch['img'], point_preds.view(1, -1, 2)]

        if 'return_polygons' in states:
            polygons = states['return_polygons']
            states['vis|polygons_train_polygons'] = [data_batch['img'], polygons]

        if 'gt_polygons' in states:
            polygons = states['gt_polygons']
            states['vis|polygons_gt_polygons'] = [data_batch['img'], polygons]

        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states
        )


        return outputs

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


    def forward(self, img, img_metas, return_loss=True, type='encode_decode', **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            if type == 'encode_decode':
                return self.forward_train_encode_decode(img, img_metas, **kwargs)
            elif type == 'dqn':
                return self.forward_train_dqn()
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train_encode_decode(self, img, img_metas, contours, contour_labels, gt_points,
                                    gt_degrees, polygons, mask, for_dqn=False, **kwargs):
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
        h_crop, w_crop = self.train_cfg.crop_size
        assert h_crop == w_crop

        B, _, H, W = img.size()
        assert B == 1
        contours = torch.tensor(contours[0], device=img.device)
        gt_points = torch.tensor(gt_points[0], device=img.device, dtype=torch.float)
        gt_degrees = torch.tensor(gt_degrees[0], device=img.device, dtype=torch.float)
        contour_labels = torch.tensor(contour_labels[0], device=img.device)
        polygons = polygons[0]
        num_degrees = gt_degrees.shape[-1]

        losses = dict()
        states = dict()

        crop_boxes = self.get_crop_boxes(H, W, h_crop, h_crop)
        num_crop = len(crop_boxes)

        crop_imgs = []
        crop_masks = []
        for crop_idx, crop_box in enumerate(crop_boxes):
            start_x, start_y, end_x, end_y = crop_box
            crop_img = img[:, :, start_y:end_y, start_x:end_x]
            crop_mask = mask[:, :, start_y:end_y, start_x:end_x]
            crop_imgs.append(crop_img)
            crop_masks.append(crop_mask)

        crop_imgs = torch.cat(crop_imgs, dim=0) # (num_crop, 3, crop_size, crop_size)
        crop_masks = torch.cat(crop_masks, dim=0) # (num_crop, 3, crop_size, crop_size)

        crop_points_list = []
        crop_gt_degrees_list = []
        crop_gt_labels_list = []
        crop_gt_inds_list = []

        # all_crop_contours = []
        # all_crop_gt_points = []
        # all_crop_gt_inds = []
        contours_assign = self.assign_points_to_boxes(contours, crop_boxes)
        gt_points_assign = self.assign_points_to_boxes(gt_points, crop_boxes)
        if len(contour_labels) > 0:
            num_patch_per_comp = torch.zeros(contour_labels.max()+1, device=img.device)

        for crop_idx, crop_box in enumerate(crop_boxes):
            start_x, start_y, end_x, end_y = crop_box
            offset = torch.tensor((start_x, start_y)).view(1, 2).to(crop_img.device)
            crop_contours = None
            if len(contours) > 0:
                # within_mask_contours = self.get_within_bbox_mask(contours, *crop_box)
                # within_mask_gt_points = self.get_within_bbox_mask(gt_points, *crop_box)
                within_mask_contours = contours_assign == crop_idx
                within_mask_gt_points = gt_points_assign == crop_idx

                mask2gt_inds = within_mask_gt_points.nonzero().view(-1)


                crop_contours = contours[within_mask_contours]
                if self.contour_noise_scale > 0:
                    noise = torch.rand(crop_contours.shape[0], 2, device=crop_contours.device)
                    crop_contours += noise * self.contour_noise_scale
                    crop_contours[:, 0] = torch.clip(crop_contours[:, 0], start_x, end_x-1)
                    crop_contours[:, 1] = torch.clip(crop_contours[:, 1], start_y, end_y-1)

                crop_contour_labels = contour_labels[within_mask_contours]
                crop_gt_points = gt_points[within_mask_gt_points]
                # all_crop_contours.append(crop_contours)
                # all_crop_gt_points.append(crop_gt_points)

                if within_mask_contours.sum() == 0 or within_mask_gt_points.sum() == 0:
                    # not gt_points within this patch
                    continue


                # all_crop_gt_inds.append(mask2gt_inds)
                num_patch_per_comp[crop_contour_labels.unique().view(-1).long()] += 1

                assign_result = self.assigner.assign(
                    crop_contours, crop_gt_points, img_metas[0]
                )
                sampling_result = self.sampler.sample(
                    assign_result,
                    crop_contours,
                    crop_gt_points,
                )
                gt_inds = sampling_result['gt_inds']
                pos_inds = sampling_result['pos_inds']
                neg_inds = sampling_result['neg_inds']
                num_points = len(pos_inds) + len(neg_inds)
                num_pos = len(pos_inds)

                if num_pos < len(crop_gt_points):
                    # not enough contours to match the gt_points, skip this patch, this should be a rare case
                    continue

                assert num_pos == len(crop_gt_points)

                crop_gt_degrees = torch.zeros(num_points, num_degrees, device=img.device, dtype=torch.float) - 1
                crop_gt_labels = torch.zeros(num_points, device=img.device, dtype=torch.long) - 1
                crop_gt_inds = torch.zeros(num_points, device=img.device, dtype=torch.long) - 1

                crop_gt_labels[:num_pos] = crop_contour_labels[pos_inds]
                crop_gt_labels[num_pos:] = crop_contour_labels[neg_inds]
                crop_gt_degrees[:num_pos] = gt_degrees[within_mask_gt_points][gt_inds[pos_inds]-1]
                crop_gt_inds[:num_pos] = mask2gt_inds[gt_inds[pos_inds] - 1]

                crop_points = sampling_result['points'].long() - offset
                norm_crop_points = self.point_head.normalize_coordinates(crop_points, w_crop, 'global')

                crop_points_list.append(crop_points + offset)
                crop_gt_degrees_list.append(crop_gt_degrees)
                crop_gt_labels_list.append(crop_gt_labels)
                crop_gt_inds_list.append(crop_gt_inds)

        if len(crop_points_list) == 0:
            if for_dqn:
                return {}
            else:
                return losses, states

        # all_crop_contours = torch.cat(all_crop_contours)
        # all_crop_gt_points = torch.cat(all_crop_gt_points)

        # if len(all_crop_contours) != len(contours):
        # assert len(all_crop_contours) == len(contours)
        # assert len(all_crop_gt_points) == len(gt_points)

        # all_crop_gt_inds = torch.cat(all_crop_gt_inds)
        # temp = gt_points.sum(dim=1).nonzero().view(-1)
        # assert set(temp.cpu().numpy()) == set(all_crop_gt_inds.unique().view(-1).cpu().numpy())

        # get contour features
        all_points = torch.cat(crop_points_list, dim=0)
        all_gt_degrees = torch.cat(crop_gt_degrees_list, dim=0)
        all_gt_labels = torch.cat(crop_gt_labels_list, dim=0)
        all_gt_inds = torch.cat(crop_gt_inds_list, dim=0)

        num_gts = len(gt_points)
        all_point_inds = torch.zeros(num_gts + 1, device=img.device, dtype=torch.long) - 1
        temp_mask = all_gt_inds >= 0
        all_point_inds[all_gt_inds[temp_mask]] = torch.arange(len(all_points), device=img.device, dtype=torch.long)[temp_mask]

        # return losses, states

        comp_idxes = all_gt_labels.unique()

        gt_next_inds = self.get_polygon_next_idx(polygons).to(img.device)
        comp_points = []
        comp_normalized_points = []
        comp_gt_inds = []
        comp_gt_labels = []
        comp_cur_point_inds = []
        comp_next_point_inds = []
        comp_edges = []

        for comp_idx in comp_idxes:
            comp_mask = all_gt_labels == comp_idx
            cur_gt_inds = all_gt_inds[comp_mask]
            pos_mask = cur_gt_inds >= 0

            if self.filter_separated_comp and num_patch_per_comp[comp_idx] > 1:
                continue

            cur_inds = all_point_inds[cur_gt_inds[pos_mask]]
            next_inds = all_point_inds[gt_next_inds[cur_gt_inds[pos_mask]]]
            # temp = gt_next_inds[cur_gt_inds[pos_mask]]
            # next_inds = cur_inds

            if len(cur_inds) > 0 and ~(next_inds == -1).any() and \
               len(cur_inds) == len(next_inds) and \
               (cur_inds.unique() == next_inds.unique()).all(): # valid components

                comp_points.append(all_points[comp_mask])
                comp_normalized_points.append(
                    self.normalize_comp_points(
                        all_points[comp_mask].float(), method=self.comp_norm_mode
                    )
                )

                comp_cur_inds = torch.zeros_like(pos_mask, dtype=torch.long) - 1
                comp_next_inds = torch.zeros_like(pos_mask, dtype=torch.long) -1
                comp_cur_inds[pos_mask] = cur_inds
                comp_next_inds[pos_mask] = next_inds

                point_ind2cur_comp_ind = torch.zeros(
                    cur_inds.max() + 1, dtype=torch.long,
                    device=img.device
                ) - 1
                row_idx = pos_mask.nonzero().view(-1)
                point_ind2cur_comp_ind[cur_inds] = row_idx
                col_idx = point_ind2cur_comp_ind[next_inds]

                comp_cur_point_inds.append(comp_cur_inds)
                comp_next_point_inds.append(comp_next_inds)
                comp_edges.append(torch.stack([row_idx, col_idx], dim=1))
                # comp_cur_point_inds.append(F.pad(cur_inds, (0, len(pos_mask) - len(cur_inds)), value=-1))
                # comp_next_point_inds.append(F.pad(next_inds, (0, len(pos_mask) - len(next_inds)), value=-1))
            else:
                pass
                # print(comp_idx)
        comp_point_window_feats = [self.get_window_feats(img[0], x, self.window_feat_size) for x in comp_points]
        if for_dqn:
            return dict(
                # point_feats_list=comp_point_feats if self.use_point_feats else comp_pred_degrees,
                point_feats_list=comp_point_window_feats,
                gt_edges_list=comp_edges,
                point_preds_list=comp_normalized_points,
                point_preds_ori_list=comp_points
            )

        num_limit_points = 512 * 512 * 8
        sizes = np.array([len(points) for points in comp_points])
        if len(sizes) == 0:
            return losses, states

        batch_idx_list, batch_size_list = self.dqn_head.greedy_arange(sizes, self.dqn_head.max_base_size)
        base_size = sizes[batch_idx_list[0][0]]
        num_max_batch = num_limit_points // base_size ** 2
        if len(batch_idx_list) > num_max_batch:
            batch_idx_list = batch_idx_list[:num_max_batch]
            batch_size_list = batch_size_list[:num_max_batch]


        batch_points = torch.zeros(
            len(batch_idx_list),
            base_size, 2, device=img.device)

        batch_normalized_points = torch.zeros(
            len(batch_idx_list),
            base_size,
            2, device=img.device)

        batch_permute_mats = torch.zeros(len(batch_idx_list),
                                        base_size,
                                        base_size,
                                        device=img.device,
                                        dtype=torch.long)

        batch_pred_degrees = torch.zeros(len(batch_idx_list),
                                         base_size,
                                         num_degrees,
                                         device=img.device,
                                         dtype=torch.float)

        batch_pred_point_cls = torch.zeros(len(batch_idx_list),
                                           base_size,
                                           2,
                                           device=img.device,
                                           dtype=torch.float)

        batch_comp_masks = torch.zeros(len(batch_idx_list),
                                       base_size,
                                       base_size,
                                       device=img.device,
                                       dtype=torch.long)

        if self.use_point_feats:
            batch_point_feats = torch.zeros(
                len(batch_idx_list),
                base_size, self.decode_head.channels, device=img.device
            )

        for i, batch_idxes in enumerate(batch_idx_list):
            cur_batch_points = torch.cat([comp_points[idx] for idx in batch_idxes])
            cur_batch_normalized_points = torch.cat([comp_normalized_points[idx] for idx in batch_idxes])
            cur_batch_pred_degrees = torch.cat([comp_pred_degrees[idx] for idx in batch_idxes])
            cur_batch_pred_point_cls = torch.cat([comp_pred_point_cls[idx] for idx in batch_idxes])
            cur_batch_cur_point_inds = torch.cat([comp_cur_point_inds[idx] for idx in batch_idxes])
            cur_batch_next_point_inds = torch.cat([comp_next_point_inds[idx] for idx in batch_idxes])
            cur_batch_comp_size = [len(comp_points[idx]) for idx in batch_idxes]
            if self.use_point_feats:
                cur_batch_point_feats = torch.cat([comp_point_feats[idx] for idx in batch_idxes])

            batch_points[i, :len(cur_batch_points)] = cur_batch_points
            batch_normalized_points[i, :len(cur_batch_normalized_points)] = cur_batch_normalized_points
            batch_pred_degrees[i, :len(cur_batch_pred_degrees)] = cur_batch_pred_degrees
            batch_pred_point_cls[i, :len(cur_batch_pred_point_cls)] = cur_batch_pred_point_cls

            if self.use_point_feats:
                batch_point_feats[i, :len(cur_batch_points)] = cur_batch_point_feats


            # prepare component mask
            num_cur_batch = len(cur_batch_points)
            pos_mask = cur_batch_cur_point_inds >= 0
            point_ind2cur_batch_ind = torch.zeros(
                cur_batch_cur_point_inds.max() + 1, dtype=torch.long,
                device=img.device
            ) - 1
            row_idx = pos_mask.nonzero().view(-1)
            point_ind2cur_batch_ind[cur_batch_cur_point_inds[pos_mask]] = row_idx
            col_idx = point_ind2cur_batch_ind[cur_batch_next_point_inds[pos_mask]]
            batch_permute_mats[i, row_idx, col_idx] = 1

            start_idx = 0
            for size in cur_batch_comp_size:
                batch_comp_masks[i, start_idx:start_idx+size, start_idx:start_idx+size] = 1
                start_idx += size

            # batch_permute_mats[i, row_idx, row_idx] = 1

            other_idx = (batch_permute_mats[i].sum(dim=-1) <= 0).nonzero().view(-1)
            if self.mask_graph:
                batch_permute_mats[i, other_idx] = -1
                batch_permute_mats[i, :, other_idx] = -1
                # temp[other_idx] = -1
                # batch_permute_mats[i, other_idx] = temp
            else:
                batch_permute_mats[i, other_idx, other_idx] = 1


        batch_pred_feats = torch.cat([batch_pred_degrees, batch_pred_point_cls], dim=-1)
        # batch_pred_feats = batch_pred_degrees

        loss_matching, state_matching = self.matching_head(
            point_feats=batch_pred_feats if not self.use_point_feats else batch_point_feats,
            graph_targets=batch_permute_mats,
            point_preds=batch_normalized_points,
            point_preds_ori=batch_points,
            comp_mask=batch_comp_masks, return_loss=True
        )
        losses.update(loss_matching)
        states.update(state_matching)

        # loss_matching, state_matching = self.dqn_head(
        # dqn_states = dict(
        #     point_feats=batch_pred_feats if not self.use_point_feats else batch_point_feats,
        #     graph_targets=batch_permute_mats,
        #     point_preds=batch_normalized_points,
        #     point_preds_ori=batch_points,
        #     comp_mask=batch_comp_masks,
        # )

        # losses.update(loss_matching)
        # states.update(state_matching)

        # if self.with_auxiliary_head:
        #     loss_aux, state_aux = self._auxiliary_head_forward_train(
        #         x, img_metas, mask)
        #     losses.update(loss_aux)
        #     states.update(state_aux)

        return losses, states, {}

    def get_proposals(self, contours, img_H, img_W, num_min_proposals=-1, device='cpu'):
        if num_min_proposals > 0:
            proposals = []
            for i, contour in enumerate(contours):
                if len(contour) < num_min_proposals:
                    add_proposals = torch.randint(low=0, high=img_H,
                                                  size=(num_min_proposals - len(contour), 2),
                                                  device=device)
                    proposals.append(torch.cat([contour, add_proposals], dim=0))
                else:
                    proposals.append(contour)
        else:
            proposals = contours

        return proposals

    def get_polygon_next_idx(self, polygons):
        poly_sizes = torch.tensor([len(polygon) for polygon in polygons], dtype=torch.long)
        num_vertices = sum(poly_sizes)

        next_idx = list(range(1, num_vertices + 1))

        offsets = []
        start_idx = 0
        for poly_size in poly_sizes:
            start_idx += poly_size.item()
            offsets.append(start_idx-1)

        next_idx = torch.tensor(next_idx, dtype=torch.long)
        # offsets = torch.tensor(offsets, dtype=torch.long).to(pos_ind.device)
        next_idx[offsets] -= poly_sizes[:len(offsets)]

        return next_idx

    def get_graph_targets(self, pos_inds, gt_inds, gt_points, polygons_list, contour_labels, num_nodes=512):

        B = len(polygons_list)
        graph_targets = []
        permute_mat = torch.zeros(B, num_nodes, num_nodes, dtype=torch.int).to(pos_inds[0].device)
        graph_mask = torch.zeros(B, num_nodes, num_nodes, dtype=torch.int).to(pos_inds[0].device)
        for i in range(B):
            polygons = polygons_list[i]
            pos_ind = pos_inds[i]
            pos2gt_ind = gt_inds[i][pos_inds[i]] - 1
            gt_point = gt_points[i]
            contour_label = contour_labels[i]
            # assert pos_ind.max() < len(contour_labels[i])
            if len(pos_ind) > 0 and pos_ind.max() >= len(contour_labels[i]):
                temp = F.pad(contour_label, (0, pos_ind.max() - len(contour_labels[i]) + 1), value=-1)
                pos_components = temp[pos_ind]
            else:
                pos_components = contour_label[pos_ind]

            poly_sizes = torch.tensor([len(polygon) for polygon in polygons], dtype=torch.long).to(pos_ind.device)
            num_vertices = sum(poly_sizes)
            num_points = len(pos_ind)
            gt2pos_ind = torch.zeros((num_points,), device=pos_ind.device, dtype=torch.long)
            gt2pos_ind[pos2gt_ind] = torch.arange(0, num_points, device=pos_ind.device, dtype=torch.long)

            if num_vertices > 0:
                next_idx = list(range(1, num_points+1))

                offsets = []
                start_idx = 0
                for poly_size in poly_sizes:
                    if start_idx + poly_size.item() > num_points:
                        next_idx[-1] = 0
                        break
                    start_idx += poly_size.item()
                    offsets.append(start_idx-1)

                next_idx = torch.tensor(next_idx, dtype=torch.long).to(pos_ind.device)
                # offsets = torch.tensor(offsets, dtype=torch.long).to(pos_ind.device)
                next_idx[offsets] -= poly_sizes[:len(offsets)]
                row_idx = torch.arange(0, num_points, dtype=torch.long, device=pos_ind.device)

                assert len(pos2gt_ind) == num_points

                # permute_mat[row_idx[gt_ind], col_idx[gt_ind]] = 1
                # if num_vertices > num_points:
                assert (next_idx >= num_points).sum() == 0
                assert pos2gt_ind.max() < num_points
                # col_idx[col_idx >= num_points] = (col_idx >= num_points).nonzero() # Form a self loop for the polygons that exceed the maximum limit
                col_idx = gt2pos_ind[next_idx[pos2gt_ind[row_idx]]]
                permute_mat[i, row_idx, col_idx] = 1

                for comp_idx in pos_components.unique():
                    if comp_idx == -1:
                        continue
                    temp = (pos_components == comp_idx) | (pos_components == -1)
                    temp2 = graph_mask[i, :num_points, :num_points][temp]
                    temp2[:, temp] = 1
                    graph_mask[i, :num_points, :num_points][temp] = temp2

            zeros_row_idx = torch.nonzero(permute_mat[i].sum(dim=-1) == 0)
            if self.mask_graph:
                permute_mat[i, zeros_row_idx, :] = -1
                permute_mat[i, :, zeros_row_idx] = -1
            else:
                permute_mat[i, zeros_row_idx, zeros_row_idx] = 1

        return permute_mat, graph_mask


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


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, contours=None, contour_labels=None,
                        comp_mask=None, **kwargs):
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
        scale_factor = self.test_scale_factor

        if scale_factor > 1:
            h_crop = h_crop // scale_factor
            w_crop = w_crop // scale_factor
            h_stride = h_stride // scale_factor
            w_stride = w_stride // scale_factor
            path_offset = torch.tensor([[1,1], [1,-1], [-1,1], [-1,-1]],
                                       device=img.device) * scale_factor * 0.5
            path_offset = path_offset.unsqueeze(0)

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
            self.init_comp_buffer(contours[i].view(-1,2), contour_labels[i].view(-1), crop_boxes)

            for crop_idx, crop_box in enumerate(tqdm(crop_boxes)):
                start_x, start_y, end_x, end_y = crop_box
                crop_img = img[i:i+1, :, start_y:end_y, start_x:end_x]
                crop_comp_mask = comp_mask[i][:, start_y:end_y, start_x:end_x]
                offset_torch = torch.tensor((start_x, start_y)).view(1, 2).to(crop_img.device)
                offset_np = np.array((start_x, start_y)).reshape(1, 2)

                crop_img = F.interpolate(crop_img, scale_factor=(scale_factor, scale_factor),
                                         mode='nearest')

                crop_comp_mask = F.interpolate(crop_comp_mask.unsqueeze(0).float(),
                                               scale_factor=(scale_factor, scale_factor),
                                               mode='nearest').int()
                # crop_comp_mask_np = (crop_comp_mask.cpu().numpy()[0,0] > 0).astype(np.uint8)
                # crop_img_np = (crop_img.cpu().numpy()[0][0] > 0).astype(np.uint8)
                # crop_img_np = crop_img.cpu()[0].permute(1,2,0).numpy()
                # mean, std, _ = img_meta[i]['img_norm_cfg'].values()
                # crop_img_np = (mmcv.imdenormalize(crop_img_np, mean=mean, std=std) > 1e-6).astype(np.uint8)[:,:,0]

                crop_img_np = (crop_comp_mask.cpu()[0].permute(1,2,0).numpy()[:,:,0] > 0).astype(np.uint8)
                cur_contours, _ = cv2.findContours(crop_img_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                if len(cur_contours) > 0:
                    cur_contours = torch.tensor(np.concatenate(cur_contours),
                                                device=crop_img.device).view(-1, 2).long()
                    cur_contour_labels = crop_comp_mask[0, 0, cur_contours[:,1], cur_contours[:,0]]
                    assert ((cur_contour_labels == 0).sum()).item() == 0
                    border_mask = ((cur_contours == 0) | (cur_contours == crop_img.size(-1) - 1)).any(dim=1)
                    cur_contours = cur_contours[~border_mask]
                    cur_contour_labels = cur_contour_labels[~border_mask]
                    cur_window_feats = self.get_window_feats(img[i], cur_contours // scale_factor + offset_torch, self.window_feat_size)

                else:
                    cur_contours = torch.zeros((0,2), device=crop_img.device)
                    cur_contour_labels = torch.zeros((0,2), dtype=torch.int, device=crop_img.device)
                    cur_window_feats = torch.zeros((0,self.window_feat_size*self.window_feat_size), device=crop_img.device)


                """
                cur_contour_labels = None
                cur_contours = None
                if contours is not None:
                    contours[i] = contours[i].view(-1, 2)
                    contour_labels[i] = contour_labels[i].view(-1)
                    if len(contours[i]) > 0:
                        # cur_contours = [vertex - offset for vertex in contours if start_x <= vertex[0] < end_x and start_y <= vertex[1] < end_y]
                        within_bbox_mask = \
                                (contours[i][:, 0] >= start_x) & \
                                (contours[i][:, 0] < end_x) & \
                                (contours[i][:, 1] >= start_y) & \
                                (contours[i][:, 1] < end_y)
                        # cur_contours = (contours[i][within_bbox_mask] - offset_torch + 0.5) * scale_factor * 1.0
                        cur_contours = (contours[i][within_bbox_mask] - offset_torch) * scale_factor * 1.0
                        cur_contour_labels = contour_labels[i][within_bbox_mask]
                        # expand the contours to four corners, this is due to the up-scaling
                        # if len(cur_contours) > 0 and scale_factor > 1:
                        #     cur_contours = (cur_contours.unsqueeze(1) + path_offset).view(-1, 2)
                        #     cur_contours[:, 0] = cur_contours[:, 0].clip(0, crop_img.shape[-1]-1)
                        #     cur_contours[:, 1] = cur_contours[:, 1].clip(0, crop_img.shape[-2]-1)
                        #     cur_contour_labels = cur_contour_labels.unsqueeze(1).repeat(1, 4).view(-1,)
                    else:
                        cur_contours = contours[i]
                        cur_contour_labels = contour_labels[i]
                """

                # seg_logit, cur_state = self.encode_decode(crop_img, [img_meta[i]])
                # feats = cur_state['feats']

                crop_polygons, crop_vertices = self.process_patch(crop_img, cur_contours,
                                                                  cur_window_feats,
                                                                  cur_contour_labels,
                                                                  offset_torch, scale_factor)
                # assert len(crop_polygons) == 1
                # assert len(crop_vertices) == 1

                # for poly_idx, temp in enumerate(crop_polygons[0]):
                #     if len(temp) > 0:
                #         crop_polygons[0][poly_idx] = crop_polygons[0][poly_idx] / scale_factor + offset_np

                # if len(crop_vertices[0]) > 0:
                #     crop_vertices[0] = crop_vertices[0] / scale_factor + offset_np

                for temp in crop_polygons:
                    cur_polygons.extend(temp)

                for temp in crop_vertices:
                    cur_vertices.append(temp)

                # if crop_idx == 1000:
                #      break

            # if len(cur_polygons) > 0:
            #     cur_polygons = np.concatenate(cur_polygons, axis=0)
            # else:
            #     cur_polygons = np.zeros((0, 2)).astype(np.int)

            cur_vertices = torch.cat(cur_vertices, dim=0)

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

        # pred_feats = torch.cat([pred_degrees, pred_point_cls], dim=-1) if not self.use_point_feats else point_feats.unsqueeze(0)
        # pred_feats = pred_degrees
        pred_feats = window_feats.unsqueeze(0)

        # pred_degrees = self.matching_head(
        #     point_feats=point_feats,
        #     return_loss=False,
        #     mode='angle'
        # ).unsqueeze(0)

        # pred_point_cls = self.matching_head(
        #     point_feats=point_feats,
        #     return_loss=False,
        #     mode='point_cls'
        # ).unsqueeze(0)

        # pred_feats = torch.cat([pred_degrees, pred_point_cls], dim=-1) if not self.use_point_feats else point_feats.unsqueeze(0)

        # cls_scores, point_preds, point_feats = self.point_det_head(
        #     img=img, feats=feats, proposals=proposals.unsqueeze(0),
        #     return_loss=False, seg_logits=seg_logits
        # )
        # pos_point_mask = cls_scores.max(dim=-1)[1][0]
        # pos_point_preds = point_preds[0, pos_point_mask][:self.num_max_test_nodes]
        # pos_point_feats = point_feats[0, pos_point_mask][:self.num_max_test_nodes]

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
            polygons, vertices = self.slide_inference(img, img_meta, rescale, contours=contours,
                                                      contour_labels=contour_labels, **kwargs)
            result['vis|polygons'] = [img, polygons]
            result['vis|points_corners'] = [img, contours]
            result['vis|points_vertices'] = [img, vertices]
            result['polygons'] = polygons
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
