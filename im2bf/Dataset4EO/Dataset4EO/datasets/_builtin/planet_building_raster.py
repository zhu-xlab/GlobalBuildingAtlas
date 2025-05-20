import os
import tarfile
import enum
import functools
import pathlib
from tqdm import tqdm
import h5py
import torch
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union
from xml.etree import ElementTree
from Dataset4EO import transforms
import pdb
import numpy as np
import math
from ..utils import clip_big_image
import geopandas as gpd
import rasterio
import shapely.geometry as shgeo
import shapely
import cv2
import json
from pycocotools.coco import COCO as COCO


from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    Zipper,
)
from torchdata.datapipes.map import Concater

from torchdata.datapipes.map import SequenceWrapper

from Dataset4EO.datasets.utils import OnlineResource, HttpResource, Dataset, ManualDownloadResource
from Dataset4EO.datasets.utils._internal import (
    path_accessor,
    getitem,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    hint_sharding,
    hint_shuffling,
    read_categories_file,
)
from Dataset4EO.features import BoundingBox, Label, EncodedImage

from .._api import register_dataset, register_info

NAME = "planet_building_raster"

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class PlanetBuildingResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://project.inria.fr/aerialimagelabeling/ and follow the instructions there.", **kwargs)


@register_dataset(NAME)
class PlanetBuildingRaster(Dataset):
    """
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "full",
        img_type = 'planet_SR',
        mask_type = 'microsoft_3m',
        skip_integrity_check: bool = True,
        crop_size =  [256, 256],
        stride: int = [192, 192],
    ) -> None:

        assert split in ('train', 'val', 'test', 'full')

        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[0,0,0], [255,255,255]]
        self.crop_size = crop_size
        self.stride=stride
        self.cat_ids = [1]
        self.cat2label = {1: 1}
        self.img_type = img_type
        self.mask_type = mask_type
        self.geo_transforms = {}
        # assert self.poly_type in ['microsoft_polygon', 'osm_polygon'], 'Invalid type of the shape file!'

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:

        city_resource = PlanetBuildingResource(
            file_name = './',
            preprocess = None,
            sha256 = None
        )

        return [city_resource]

    def _parse_ann_info(self, img_infos):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        res_dict = dict()
        for key, img_info in img_infos.items():

            img_id = img_info['id']
            coco = self.gdal_coco if key == 'gdal_info' else self.gt_coco
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            ann_info = coco.loadAnns(ann_ids)

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_masks_ann = []
            features = []
            for i, ann in enumerate(ann_info):
                if ann.get('ignore', False):
                    continue
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
                inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                """
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                """
                if ann['category_id'] not in self.cat_ids:
                    continue
                features.append(ann['features'])
                bbox = [x1, y1, x1 + w, y1 + h]
                if ann.get('iscrowd', False):
                    gt_bboxes_ignore.append(bbox)
                else:
                    gt_bboxes.append(bbox)
                    gt_labels.append(self.cat2label[ann['category_id']])
                    gt_masks_ann.append(ann.get('segmentation', None))

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

            if gt_bboxes_ignore:
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            else:
                gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

            # seg_map = img_info['filename'].replace('jpg', 'png')
            seg_map = None

            ann = dict(
                bboxes=gt_bboxes,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore,
                # masks=gt_masks_ann,
                features=features,
                seg_map=seg_map)

            res = dict(
                filename=img_info['filename'],
                maskname=img_info['mask_name'],
                img_id=img_id,
                ann=ann,
                height=img_info['height'],
                width=img_info['width']
            )

            res_dict[key] = res

        return res_dict

    def _classify_dp(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('.tif'):
            return 0
        return 1

    def get_image_infos(self, crop_boxes, out_dir, img_name, mask_name, mask_scale):
        images = []
        for img_id, crop_box in enumerate(crop_boxes):
            start_x, start_y, end_x, end_y = crop_box
            crop_img_name = img_name + f'_{start_x}x{start_y}'
            crop_img_path = os.path.join(out_dir, f'{self.img_type}_dir', crop_img_name + '.png')
            crop_mask_name = mask_name + f'_{start_x*mask_scale}x{start_y*mask_scale}'
            crop_mask_path = os.path.join(out_dir, f'{self.mask_type}_dir', crop_mask_name + '.png')

            img_dict = dict(
                id=img_id,
                width=int(end_x - start_x),
                height=int(end_y - start_y),
                file_name=crop_img_path,
                mask_name=crop_mask_path
            )
            images.append(img_dict)
        return images

    def split_images(self, image, crop_boxes, out_dir, img_name):

        image = np.transpose(image, axes=(1,2,0))
        if image.dtype == np.uint16:
            image = (image // 20).clip(0, 255).astype(np.uint8)
        elif img_name.endswith('our_new') or img_name.endswith('microsoft3_new') or img_name.endswith('osm3_new'): # building footprint
            image[image == 1] = 255

        for img_idx, crop_box in tqdm(enumerate(crop_boxes)):
            start_x, start_y, end_x, end_y = crop_box
            crop_img_name = img_name + f'_{start_x}x{start_y}'
            crop_img_path = os.path.join(out_dir, crop_img_name + '.png')

            cropped_image = image[start_y:end_y, start_x:end_x, ...]
            cv2.imwrite(crop_img_path, cropped_image)

    def split_polygons(self, shp_file, transform, crop_boxes, H, W, iou_thre=0.4):
        ## Extract all the polygons
        def get_bounds(polygon):
            vertice_x = [vertice[0] for vertice in polygon]
            vertice_y = [vertice[1] for vertice in polygon]
            return min(vertice_x), min(vertice_y), max(vertice_x), max(vertice_y)

        def parse_geometry(geom, transform):

            def parse_polygon(polygon, transform):
                pixel_exterior = np.array(
                    [~transform * (x, y) for x, y in polygon.exterior.coords]
                )
                pixel_interiors = []
                for ring in polygon.interiors:
                    pixel_interiors.append(
                        np.array([~transform * (x, y) for x, y in ring.coords]).tolist()
                    )

                json_polygon = dict(
                    exterior=pixel_exterior.tolist(),
                    interiors=pixel_interiors,
                    area=polygon.area
                )
                return json_polygon

            results = []
            if geometry.geom_type == "MultiPolygon":
                for polygon in geom.geoms:
                    json_polygon = parse_polygon(polygon, transform)
                    results.append(json_polygon)

            elif geometry.geom_type == "Polygon":
                results.append(parse_polygon(geometry, transform))

            else:
                return False, None

            return True, results

        def get_within_feature_ids(crop_box, bounds):
            start_x, start_y, end_x, end_y = crop_box
            flag1 = bounds[:,0] >= start_x
            flag2 = bounds[:,2] < end_x
            flag3 = bounds[:,1] >= start_y
            flag4 = bounds[:,3] < end_y

            return flag1 & flag2 & flag3 & flag4
 
        # polygons = []
        bounds = []
        features = []
        for index, row in shp_file.iterrows():
            geometry = row['geometry']
            cur_bound = np.array(geometry.bounds).reshape(2, 2)
            cur_pixel_bound = np.array([~transform * (x, y) for x, y in cur_bound])
            flag, json_geom = parse_geometry(geometry, transform)
            if flag:
                features.append(json_geom)
                bounds.append(cur_pixel_bound.reshape(-1))

        bounds = np.stack(bounds, axis=0)
        bounds = bounds[:,[0,3,2,1]]

        ann_id = 0
        annotations = []
        for img_id, crop_box in tqdm(enumerate(crop_boxes)):
            start_x, start_y, end_x, end_y = crop_box
            # shape_box = shgeo.Polygon([(start_x, start_y), (end_x, start_y), (end_x, end_y), (start_x, end_y)])

            ids = get_within_feature_ids(crop_box, bounds)
            ids = list(np.where(ids)[0])
            ## saved cropped image
            for idx in ids:
                minx, miny, maxx, maxy = bounds[idx].tolist()
                w, h = maxx-minx, maxy-miny
                bbox = [minx + w//2 - start_x, miny + h//2 - start_y, w, h]
                annotation = {
                    "id": ann_id,
                    "category_id": 1,  # Building
                    "bbox": [round(x, 2) for x in bbox],
                    # "segmentation": self.sub_polygon(polygon, start_x, start_y)[:-1],
                    "features": self.add_offset_to_features(features[idx], (start_x, start_y)),
                    "score": 1.0,
                    "image_id": img_id,
                    # "area": shgeo.Polygon(polygon).area
                }
                annotations.append(annotation)
                ann_id += 1

        return annotations

    def aligned_ann(self, coco):
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            if len(ann_ids) > 0:
                info = self.coco.loadImgs([i])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
                total_ann_ids.extend(ann_ids)



    def _datapipe(self, resource_dp):

        raster_dp, _ = Demultiplexer(resource_dp[0], 2, self._classify_dp)
        raster_paths = list(raster_dp)
        self.raster_paths = raster_paths

        for raster_path in raster_paths:
            path = raster_path[0]
            img_tif = rasterio.open(path)
            self.geo_transforms[pathlib.Path(path).name] = img_tif.transform

        return [x[0] for x in raster_paths]

        # print(f'clipping the polygon for {city_name}...')

        geom_dict = {}
        for raster_path in raster_paths:
            path = raster_path[0]

            img_tif = rasterio.open(path)
            self.geo_transforms[pathlib.Path(path).name] = img_tif.transform

            geometries = []
            for index, row in shp_file.iterrows():
                geometry = row['geometry']

                if geometry.geom_type == 'Polygon' and geometry.is_valid:
                    cur_bound = np.array(geometry.bounds).reshape(2, 2)
                    geometries.append(geometry)

            lens = [g.length for g in geometries]

            geom_dict[pathlib.Path(path).name] = geometries

        self.geom_dict = geom_dict

        return geom_dict

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def add_offset_to_features(self, features, offset):
        np_offset = np.array(offset).reshape(1,2)
        new_features = []
        for polygon in features:
            new_polygon = dict(
                exterior=(np.array(polygon['exterior']) - np_offset).tolist(),
                interiors=[
                    (np.array(x).reshape(-1,2) - np_offset).tolist() for x in polygon['interiors']
                ]
            )
            new_features.append(new_polygon)

        return new_features

    def get_crop_boxes(self, img_H, img_W, crop_size=(256, 256), stride=(192, 192)):
        # prepare locations to crop
        if crop_size[0] <= 0 or crop_size[1] <= 0:
            return np.array([[0, 0, img_W, img_H]])

        num_rows = math.ceil((img_H - crop_size[0]) / stride[0]) if \
            math.ceil((img_H - crop_size[0]) / stride[0]) * stride[0] + crop_size[0] >= img_H \
            else math.ceil( (img_H - crop_size[0]) / stride[0]) + 1

        num_cols = math.ceil((img_W - crop_size[1]) / stride[1]) if math.ceil(
            (img_W - crop_size[1]) /
            stride[1]) * stride[1] + crop_size[1] >= img_W else math.ceil(
                (img_W - crop_size[1]) / stride[1]) + 1

        x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
        xmin = x * stride[1]
        ymin = y * stride[0]

        xmin = xmin.ravel()
        ymin = ymin.ravel()
        xmin_offset = np.where(xmin + crop_size[1] > img_W, img_W - xmin - crop_size[1],
                               np.zeros_like(xmin))
        ymin_offset = np.where(ymin + crop_size[0] > img_H, img_H - ymin - crop_size[0],
                               np.zeros_like(ymin))
        boxes = np.stack([
            xmin + xmin_offset, ymin + ymin_offset,
            np.minimum(xmin + crop_size[1], img_W),
            np.minimum(ymin + crop_size[0], img_H)
        ], axis=1)

        return boxes

    def __len__(self) -> int:

        return len(self.raster_paths)

