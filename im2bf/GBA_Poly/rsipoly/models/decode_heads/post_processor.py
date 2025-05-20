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
import shapely
import cv2

from ..builder import HEADS, build_head


@HEADS.register_module()
class PostProcessor(BaseModule):
    def __init__(self, filter_iou_thre=0.7, **kwargs):
        super(PostProcessor, self).__init__(**kwargs)
        self.filter_iou_thre = filter_iou_thre

    def _poly2mask(self, poly, img_size):
        mask = np.zeros(img_size, dtype=np.uint8)
        pdb.set_trace()
        # polygon_coords = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(mask, poly, color=1)
        return mask

    def get_boundary(self, poly):
        start_x, start_y = poly.min(axis=0)
        end_x, end_y = poly.min(axis=0)

        return np.array([start_x, start_y, end_x, end_y])


    def filter_by_iou(self, mask, polygons, thre):
        B, C, H, W = mask.shape
        num_poly = len(polygons)
        bounds = [self.get_boundary(poly) for poly in polygons]
        bounds = np.stack(bounds, axis=0)
        is_valid = np.zeros(num_poly, dtype=np.int8)+1
        for i in range(num_poly):
            start_x, start_y = polygons[i].max(axis=0)
            end_x, end_y = polygons[i].max(axis=0)
            bound_idx = (bounds[:, 0] >= end_x) | (bounds[:, 1] <= start_x) \
                    | (bounds[:, 2] >= end_y) | (bounds[:, 3] < start_y)
            bound_idx = (~ bound_idx) & (np.arange(num_poly) > i)
            bound_idx = np.nonzero(bound_idx)[0]
            if len(bound_idx) > 0:
                pdb.set_trace()
                poly_i = shapely.Polygon(polygons[i])
                for j in range(len(bound_idx)):
                    poly_i = shapely.Polygon(polygons[j])

                    intersect = poly_j.intersection(poly_i)
                    if not intersect.is_empty and intersect.geom_type == 'Polygon':
                        iou = intersect.area / (poly_j.area + 1e-8)
                        if iou > thre:
                            is_valid[j] = False


        # poly_masks = [self._poly2mask(poly, (H, W)) for poly in polygons]



    def post_process(self, img, polygons):
        B, C, H, W = img.shape
        for i in range(B):
            if self.filter_iou_thre > 0:
                polygons = self.filter_by_iou(img, polygons[i], self.filter_iou_thre)
