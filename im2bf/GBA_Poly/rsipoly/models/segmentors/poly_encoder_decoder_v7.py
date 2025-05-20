# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import numpy as np
import mmcv

from rsipoly.core import add_prefix
from rsipoly.ops import resize
from .. import builder
from ..builder import SEGMENTORS
# from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from mmcv.parallel import DataContainer
from pycocotools import mask as cocomask


@SEGMENTORS.register_module()
class PolyEncoderDecoderV7(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, backbone, decode_head, point_det_head, matching_head, suppression,
                 num_degrees=36, wandb_cfg=None, **kwargs):
        super(PolyEncoderDecoderV7, self).__init__(backbone, decode_head, **kwargs)
        self.matching_head = builder.build_head(matching_head)
        self.suppression = builder.build_head(suppression)
        self.point_det_head = builder.build_head(point_det_head)
        self.wandb_cfg = wandb_cfg
        self.num_max_test_nodes = 1024
        self.prob_thre = 0.0
        self.mask_graph = False
        self.nms_width = 7
        self.num_min_proposals = 512
        self.num_degrees = num_degrees
        if self.test_cfg:
            self.num_max_test_nodes = self.test_cfg.get('num_max_test_nodes', 1024)
            self.prob_thre = self.test_cfg.get('prob_thre', 0.0)
            self.nms_width = self.test_cfg.get('nms_width', 7)

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
        losses, states = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        # states.update({'img': data_batch['img'], 'gt': data_batch['mask']})
        states.update({'img': data_batch['img']})
        states['vis|masks_gt'] = [data_batch['img'], data_batch['mask'][:, 0].cpu().numpy()]

        point_preds = self.matching_head.normalize_coordinates(states['point_preds'].detach(), W, input='normalized')
        probs = F.softmax(states['seg_logits'], dim=1)[:, 1:, ...].detach()
        states['vis|featmap_probs'] = [data_batch['img'], probs]
        states['vis|points_vertices'] = [data_batch['img'], data_batch['contours']]
        # states['vis|points_pos_point_preds'] = [data_batch['img'], states['pos_point_preds']]
        states['vis|points_point_preds'] = [data_batch['img'], point_preds]
        if 'return_polygons' in states:
            polygons = states['return_polygons']
            states['vis|polygons_train_polygons'] = [data_batch['img'], polygons]

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=states)

        return outputs

    def forward_train(self, img, img_metas, contours, contour_labels, gt_points, polygons, mask,
                      gt_degrees, **kwargs):
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
        contour_labels = [torch.tensor(contour_label).to(img.device) for contour_label in contour_labels]

        # gt_labels = [torch.ones(len(gt_point)).to(img.device).to(torch.int) for gt_point in gt_points]

        x = self.extract_feat(img)

        losses = dict()
        states = dict()

        # seg_logits, state_decode = self.decode_head.forward(x)
        # states.update(state_decode)

        loss_decode, state_decode = self._decode_head_forward_train(x, img_metas, mask)
        losses.update(loss_decode)
        states.update(state_decode)
        feats = state_decode['feats']
        seg_logits = states['seg_logits']

        _, _, H, W = img.shape
        assert H == W # This make things easier
        proposals = self.get_proposals(contours, H, W, num_min_proposals=self.num_min_proposals, device=img.device)

        # def forward(self, img, feats, proposals, gt_points, gt_labels):

        loss_det, state_det = self.point_det_head(
            img=img, feats=feats, proposals=proposals,
            gt_points=gt_points, seg_logits=seg_logits,
            img_metas=img_metas
        )
        losses.update(loss_det)
        states.update(state_det)

        point_preds = state_det['point_preds']
        point_feats = state_det['point_feats']
        gt_inds = state_det['gt_inds']
        pos_inds = state_det['pos_inds']
        labels = state_det['labels']

        graph_targets, degree_mat, comp_mask = self.get_graph_targets(pos_inds, gt_inds, gt_points, polygons,
                                                                      contour_labels=contour_labels,
                                                                      degrees_list=gt_degrees)

        # merged_vertices, merged_vertices_mask, merged_permute_matrix = \
        #         self.merge_preds_gts(pred_points, vertices, vertices_mask, permute_matrix)

        # pos_point_feats = [point_feat[label > 0] for point_feat, label in zip(point_feats, labels)]
        loss_matching, state_matching = self.matching_head(
            img=img, point_feats=point_feats, graph_targets=graph_targets, point_preds=point_preds,
            comp_mask=comp_mask, gt_angles=degree_mat, return_loss=True
        )
        losses.update(loss_matching)
        states.update(state_matching)


        # if self.with_auxiliary_head:
        #     loss_aux, state_aux = self._auxiliary_head_forward_train(
        #         x, img_metas, mask)
        #     losses.update(loss_aux)
        #     states.update(state_aux)

        return losses, states

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

    def get_graph_targets(self, pos_inds, gt_inds, gt_points, polygons_list, contour_labels,
                          degrees_list, num_nodes=512):

        B = len(polygons_list)
        graph_targets = []
        permute_mat = torch.zeros(B, num_nodes, num_nodes, dtype=torch.int).to(pos_inds[0].device)
        degree_mat = torch.zeros(B, num_nodes, self.num_degrees, dtype=torch.int).to(pos_inds[0].device)
        graph_mask = torch.zeros(B, num_nodes, num_nodes, dtype=torch.int).to(pos_inds[0].device)
        bins = torch.linspace(0, 2 * torch.pi, self.num_degrees, device=pos_inds[0].device)
        for i in range(B):
            polygons = polygons_list[i]
            degrees = degrees_list[i]
            degrees = [torch.tensor(degree, device=pos_inds[0].device) for degree in degrees]
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
                degrees = torch.cat(degrees)
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
                next_idx[offsets] -= poly_sizes[:len(offsets)]
                row_idx = torch.arange(0, num_points, dtype=torch.long, device=pos_ind.device)

                assert len(pos2gt_ind) == num_points

                assert (next_idx >= num_points).sum() == 0
                assert pos2gt_ind.max() < num_points
                col_idx = gt2pos_ind[next_idx[pos2gt_ind[row_idx]]]
                permute_mat[i, row_idx, col_idx] = 1
                indices = torch.bucketize(degrees, bins)
                degree_one_hot = torch.zeros(len(indices), self.num_degrees, device=pos_ind.device,
                                             dtype=torch.int)
                degree_one_hot.scatter_(1, indices.view(-1, 1), 1)
                degree_mat[i, :num_points] = degree_one_hot[pos2gt_ind[row_idx]]

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

        return permute_mat, degree_mat, graph_mask


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
    def slide_inference(self, img, img_meta, rescale, contours=None, contour_labels=None):
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
                        cur_contours = contours[i][within_bbox_mask] - offset_torch
                        cur_contour_labels = contour_labels[i][within_bbox_mask]
                    else:
                        cur_contours = contours[i]
                        cur_contour_labels = contour_labels[i]


                # seg_logit, cur_state = self.encode_decode(crop_img, [img_meta[i]])
                # feats = cur_state['feats']

                crop_polygons, crop_vertices = self.get_polygons(crop_img, cur_contours,
                                                                 cur_contour_labels)
                assert len(crop_polygons) == 1
                assert len(crop_vertices) == 1

                for poly_idx, temp in enumerate(crop_polygons[0]):
                    if len(temp) > 0:
                        crop_polygons[0][poly_idx] += offset_np

                if len(crop_vertices[0]) > 0:
                    crop_vertices[0] += offset_np

                cur_polygons.extend(crop_polygons[0])
                cur_vertices.append(crop_vertices[0])
                if img_id == 500:
                     break

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

    def get_polygons(self, img, contours, contour_labels):

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

        cls_scores, point_preds, point_feats = self.point_det_head(
            img=img, feats=feats, proposals=proposals.unsqueeze(0),
            return_loss=False, seg_logits=seg_logits
        )
        # pos_point_mask = cls_scores.max(dim=-1)[1][0]
        # pos_point_preds = point_preds[0, pos_point_mask][:self.num_max_test_nodes]
        # pos_point_feats = point_feats[0, pos_point_mask][:self.num_max_test_nodes]

        # point_preds_ori = self.matching_head.normalize_coordinates(point_preds, W, 'normalized')
        point_preds_ori = proposals.unsqueeze(0)
        bbox_preds = self.generate_bbox_from_point(point_preds_ori[0], width=self.nms_width)
        probs = F.softmax(cls_scores, dim=-1)[0, :, 1]
        keep_idx = mmcv.ops.nms(bbox_preds, probs, iou_threshold=0.3)[1].cpu().numpy()
        prob_idx = torch.nonzero(probs > self.prob_thre).squeeze(1).cpu().numpy()
        keep_idx = np.intersect1d(keep_idx, prob_idx)
        if len(keep_idx) > 0:
            # keep_idx = torch.randperm(len(probs), device=probs.device)[:self.num_max_test_nodes]
            # pos_point_preds = point_preds_ori[0, keep_idx[:self.num_max_test_nodes]]
            pos_point_preds = point_preds[0, keep_idx[:self.num_max_test_nodes]]
            # pos_point_preds = proposals[keep_idx[:self.num_max_test_nodes]]
            # pos_point_preds = self.matching_head.normalize_coordinates(pos_point_preds, W, 'normalized')
            pos_point_feats = point_feats[0, keep_idx[:self.num_max_test_nodes]]
            pos_contour_labels = contour_labels[keep_idx[:self.num_max_test_nodes]]
            comp_mask = self.get_comp_mask(pos_contour_labels)

            polygons, vertices = self.matching_head(
                img=img, point_feats=pos_point_feats.unsqueeze(0),
                point_preds=pos_point_preds.unsqueeze(0),
                comp_mask=comp_mask.unsqueeze(0),
                return_loss=False
            )
            return polygons, vertices
        else:
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



    def inference(self, img, img_meta, rescale, contours=None, contour_labels=None):
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
                                                      contour_labels=contour_labels)
            result['vis|polygons'] = [img, polygons]
            result['vis|points_corners'] = [img, contours]
            result['vis|points_vertices'] = [img, vertices]
            result['polygons'] = polygons
        else:
            seg_logit, state = self.whole_inference(img, img_meta, rescale)
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


    def simple_test(self, img, img_meta, contours=None, contour_labels=None, rescale=True):
        """Simple test with single image."""
        results = self.inference(img, img_meta, rescale, contours=contours, contour_labels=contour_labels)
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
