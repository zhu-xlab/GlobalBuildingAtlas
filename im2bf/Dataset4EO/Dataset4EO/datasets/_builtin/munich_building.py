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
    Concater,
    Zipper,
)

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

NAME = "munich_building"
_TRAIN_LEN = 31 * 5 * 25
_VAL_LEN = 5 * 5 * 25
_TEST_LEN = 36 * 5 * 25

_TRAIN_CITY_NAMES = ['austin{}.tif', 'chicago{}.tif', 'kitsap{}.tif', 'vienna{}.tif', 'tyrol-w{}.tif']

_VAL_IMG_NAMES = []

for idx in range(1, 6, 1):
    for name in _TRAIN_CITY_NAMES:
        _VAL_IMG_NAMES.append(name.format(idx))


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class MunichBuildingResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://project.inria.fr/aerialimagelabeling/ and follow the instructions there.", **kwargs)


@register_dataset(NAME)
class MunichBuilding(Dataset):
    """
    - **homepage**: https://project.inria.fr/aerialimagelabeling/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        city_names=None, # None -> use all cities
        data_info: bool = True,
        skip_integrity_check: bool = True,
        crop_size: int = 256,
        stride: int = 192
    ) -> None:

        assert split in ('train', 'val', 'train_val', 'test')

        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[0,0,0], [255,255,255]]
        self.data_info = data_info
        self.crop_size = crop_size
        self.stride=stride
        self.city_names = city_names
        self.cat_ids = [1]
        self.cat2label = {1: 1}

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        mask_resource = MunichBuildingResource(
            file_name = 'munich_our_new.tif',
            preprocess = None,
            sha256 = None
        )
        shp_resource = MunichBuildingResource(
            file_name = 'munich_microsoft.shp',
            preprocess = None,
            sha256 = None
        )

        return [mask_resource, shp_resource]

    def _parse_ann_info(self, img_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        img_id = img_info['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
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
            masks=gt_masks_ann,
            seg_map=seg_map)

        res = dict(
            filename=img_info['filename'],
            img_id=img_id,
            ann=ann
        )

        return res

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        clip_dir = self._clip_images(resource_dps)
        self.coco = COCO(os.path.join(clip_dir, 'poly_ann.json'))

        img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"

        dp = Mapper(iter(data_infos), self._parse_ann_info)
        return dp

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

    def sub_polygon(self, polygon, start_x, start_y):
        sub_polygon = []
        for (x, y) in polygon:
            sub_polygon.append((x - start_x, y - start_y))

        return sub_polygon


    def clip_big_image_ann(self, img_path, gt_path, save_dir, clip_size=256, stride_size=192, iou_thre=0.3):
        shapefile = gpd.read_file(gt_path)
        tiff = rasterio.open(img_path)
        img_name = img_path.split('/')[-1].split('.')[0]
        transform = tiff.transform
        H, W = tiff.shape
        image = np.transpose(tiff.read(), axes=(1,2,0))

        # split the large image
        num_rows = math.ceil((H - clip_size) / stride_size) if math.ceil(
            (W - clip_size) /
            stride_size) * stride_size + clip_size >= H else math.ceil(
                (H - clip_size) / stride_size) + 1
        num_cols = math.ceil((W - clip_size) / stride_size) if math.ceil(
            (W - clip_size) /
            stride_size) * stride_size + clip_size >= W else math.ceil(
                (W - clip_size) / stride_size) + 1

        x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
        xmin = x * stride_size
        ymin = y * stride_size

        xmin = xmin.ravel()
        ymin = ymin.ravel()
        xmin_offset = np.where(xmin + clip_size > W, W - xmin - clip_size,
                               np.zeros_like(xmin))
        ymin_offset = np.where(ymin + clip_size > H, H - ymin - clip_size,
                               np.zeros_like(ymin))
        boxes = np.stack([
            xmin + xmin_offset, ymin + ymin_offset,
            np.minimum(xmin + clip_size, W),
            np.minimum(ymin + clip_size, H)
        ], axis=1)

        # Rescale geo-coordinated polygons to be with pixel coordinates
        polygons = []
        ranges = []
        for index, row in shapefile.iterrows():
            geometry = row['geometry']
            if geometry.geom_type == "MultiPolygon":
                for polygon in geometry.geoms:
                    polygons.append(polygon)
            elif geometry.geom_type == "Polygon":
                polygons.append(geometry)
            else:
                raise TypeError(f"geometry.type should be either Polygon or MultiPolygon, not {geometry.type}.")

        images = []
        annotations = []
        img_id = 0
        ann_id = 0
        for box in tqdm(boxes):
            start_x, start_y, end_x, end_y = box
            sub_img_name = img_name + f'_{start_x}x{start_y}'
            sub_img_path = os.path.join(save_dir, 'mask_dir', sub_img_name + '.png')

            scaled_coordinates = []
            scaled_polygons = []
            shape_box = shgeo.Polygon([(start_x, start_y), (end_x, start_y),
                                      (end_x, end_y), (start_x, end_y)])

            ## Create image dict
            img_dict = dict(
                id=img_id,
                width=clip_size,
                height=clip_size,
                file_name=sub_img_path,
            )
            images.append(img_dict)
            ## saved cropped image
            cropped_image = image[start_y:end_y, start_x:end_x, ...]
            cv2.imwrite(sub_img_path, cropped_image)

            has_building = False
            for polygon in polygons:
                coords = np.array(polygon.exterior.coords)
                pixel_coords = [~transform * (x, y) for x, y in coords]
                scaled_polygon = [(round(x / W * W, 2), round(y / H * H, 2)) for x, y in pixel_coords]
                shape_contour = shgeo.Polygon(scaled_polygon)

                try:
                    inter_poly, half_iou = self.calchalf_iou(shape_contour, shape_box)
                except shapely.errors.GEOSException as e:
                    print(e)
                    pass

                if half_iou > 1 - 1e-8:
                    minx, miny, maxx, maxy = shape_contour.bounds
                    w, h = maxx-minx, maxy-miny
                    bbox = [minx + w//2 - start_x, miny + h//2 - start_y, w, h]
                    annotation = {
                        "id": ann_id,
                        "category_id": 1,  # Building
                        "bbox": [round(x, 2) for x in bbox],
                        "segmentation": self.sub_polygon(scaled_polygon, start_x, start_y)[:-1],
                        "score": 1.0,
                        "image_id": img_id,
                        "area": shgeo.Polygon(scaled_polygon).area
                    }
                    annotations.append(annotation)
                    ann_id += 1
                    has_building = True

                elif half_iou > iou_thre:
                    minx, miny, maxx, maxy = inter_poly.bounds
                    w, h = maxx-minx, maxy-miny
                    # bbox = [minx + w//2, miny + h//2, w, h]
                    bbox = [minx + w//2 - start_x, miny + h//2 - start_y, w, h]
                    has_building = True

                    if inter_poly.geom_type  == 'Polygon':
                        # inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                        out_poly = list(inter_poly.exterior.coords)
                        out_poly = [(round(x, 2), round(y, 2)) for x, y in out_poly]
                        annotation = {
                            "id": ann_id,
                            "category_id": 1,  # Building
                            "bbox": [round(x, 2) for x in bbox],
                            "segmentation": self.sub_polygon(out_poly, start_x, start_y)[:-1],
                            "score": 1.0,
                            "image_id": img_id,
                            "area": shgeo.Polygon(out_poly).area
                        }
                        annotations.append(annotation)
                        ann_id += 1

                    elif inter_poly.geom_type == 'MultiPolygon':
                        for single_poly in inter_poly.geoms:
                            single_poly = list(single_poly.exterior.coords)
                            out_poly = [(round(x, 2), round(y, 2)) for x, y in single_poly]
                            annotation = {
                                "id": ann_id,
                                "category_id": 1,  # Building
                                "bbox": [round(x, 2) for x in bbox],
                                "segmentation": self.sub_polygon(out_poly, start_x, start_y)[:-1],
                                "score": 1.0,
                                "image_id": img_id,
                                "area": shgeo.Polygon(out_poly).area
                            }
                            annotations.append(annotation)
                            ann_id += 1
                    else:
                        # Could be GeometryCollection
                        pass

            if img_id == 10:
                break

            img_id += 1

        ann_dict = dict(
            images=images,
            categories=[{"id": 1, "name": 'building'}],
            annotations=annotations
        )
        with open(os.path.join(save_dir, 'poly_ann.json'), 'w') as f:
            f.write(json.dumps(ann_dict))

    def _clip_images(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        mask_dp, shp_dp = resource_dps

        clip_size=self.crop_size
        stride_size=self.stride
        # img_W = 5000

        # assert clip_size < img_W and clip_size > 0
        # assert img_W > stride_size and stride_size > 0
        # num_imgs = math.ceil((img_W - clip_size) / stride_size) + 1

        # dp = eval(f'{split}_dp')

        # img_dp, gt_dp = Demultiplexer(
        #     dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        # )

        clip_dir = os.path.join(self.root,
                                'clip_c{}_s{}'.format(clip_size, stride_size))

        clip_dir_img = os.path.join(self.root,
                                    'clip_c{}_s{}'.format(clip_size, stride_size),
                                    'mask_dir')

        if os.path.exists(clip_dir_img): # images already generated
            pass
        else:
            os.makedirs(clip_dir_img)
            print(f'Clipping the dataset to patches of sizes {clip_size} x {clip_size}. May take a while...')
            for img_path, gt_path in tqdm(zip(mask_dp, shp_dp)):
                img_path = img_path[0]
                gt_path = gt_path[0]

                self.clip_big_image_ann(img_path, gt_path, save_dir=clip_dir, clip_size=clip_size, stride_size=stride_size)

        return clip_dir


    def __len__(self) -> int:
        base_len = {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'train_val': _TRAIN_LEN + _VAL_LEN,
            'test': _TEST_LEN

        }[self._split]
        if self.city_names is not None:
            base_len = int(base_len / 5 * len(self.city_names))

        return base_len

if __name__ == '__main__':
    dp = Landslide4Sense('./')
