# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import pdb

import mmcv
import cv2
import numpy as np
import copy

from ..builder import PIPELINES
from .compose import Compose

def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {
        'gt_bboxes': 'gt_labels',
        'gt_bboxes_ignore': 'gt_labels_ignore'
    }
    bbox2mask = {
        'gt_bboxes': 'gt_masks',
        'gt_bboxes_ignore': 'gt_masks_ignore'
    }
    bbox2seg = {
        'gt_bboxes': 'gt_semantic_seg',
    }
    return bbox2label, bbox2mask, bbox2seg

@PIPELINES.register_module()
class RandomRotate90:

    def __init__(self, prob=0.75):
        self.prob = prob

    def _rotate_bboxes(self, results, rotate_matrix):

        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = np.stack([[min_x, min_y], [max_x, min_y],
                                    [min_x, max_y],
                                    [max_x, max_y]])  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (coordinates,
                 np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                axis=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose(
                (2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix,
                                       coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = np.min(
                rotated_coords[:, :, 0], axis=1), np.min(
                    rotated_coords[:, :, 1], axis=1)
            max_x, max_y = np.max(
                rotated_coords[:, :, 0], axis=1), np.max(
                    rotated_coords[:, :, 1], axis=1)
            min_x, min_y = np.clip(
                min_x, a_min=0, a_max=w), np.clip(
                    min_y, a_min=0, a_max=h)
            max_x, max_y = np.clip(
                max_x, a_min=min_x, a_max=w), np.clip(
                    max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y],
                                    axis=-1).astype(results[key].dtype)

    def _rotate_features(self, results, rotate_matrix):
        new_features = []
        for feature in results['features']:
            new_feature = feature.copy()
            new_rings = []
            for ring in feature['coordinates']:
                N = len(ring)
                temp = np.concatenate([ring, np.ones((N, 1))], axis=-1)
                new_ring = np.matmul(temp, rotate_matrix.transpose(1,0))
                new_rings.append(new_ring.tolist())

            new_feature['coordinates'] = new_rings
            new_features.append(new_feature)

        results['features'] = new_features

    def __call__(self, results):

        if np.random.rand() < self.prob:
            rot_k = np.random.choice([0,1,2,3])
            angle = [0, -90, 180, 90][rot_k]

            results['rotate_k'] = rot_k

            h, w = results['img'].shape[:2]
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            self._rotate_bboxes(results, rotate_matrix)

            for key in results.get('img_fields', ['img']):
                # img = results[key].copy()
                # img_rotated = mmcv.imrotate(
                #     img, angle, center, scale, border_value=self.img_fill_val)
                results[key] = np.rot90(results[key], k=rot_k, axes=(0,1)).copy()
                # results[key] = img_rotated.astype(img.dtype)
                results['img_shape'] = results[key].shape

            h, w, c = results['img_shape']
            for key in results.get('mask_fields', []):
                masks = results[key]
                # results[key] = masks.rotate((h, w), angle, center, scale, fill_val)
                results[key] = np.rot90(masks, k=rot_k, axes=(0,1)).copy()

            for key in results.get('seg_fields', []):
                # seg = results[key].copy()
                # results[key] = mmcv.imrotate(
                #     seg, angle, center, scale,
                #     border_value=fill_val).astype(seg.dtype)
                results[key] = np.rot90(results[key], k=rot_k, axes=(0,1)).copy()

            if 'features' in results:
                self._rotate_features(results, rotate_matrix)
                # results['feats'] = np.rot90(results['feats'], k=rot_k, axes=(0,1)).copy()

        return results
