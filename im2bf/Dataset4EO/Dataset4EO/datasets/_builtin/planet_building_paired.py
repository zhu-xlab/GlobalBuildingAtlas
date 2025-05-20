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

NAME = "planet_building_paired"
_TRAIN_LEN = 31 * 5 * 25
_VAL_LEN = 5 * 5 * 25
_TEST_LEN = 36 * 5 * 25
_POSTFIX_MAP = {
    'SR.tif': 'planet_SR',
    'our_new.tif': 'building_footprint',
    'regu-v1.tif': 'building_footprint_regu-v1',
    'regu-v2.tif': 'building_footprint_regu-v2',
    'regu-v3.tif': 'building_footprint_regu-v3',
    'microsoft3_new.tif': 'microsoft_3m',
    'osm_new.shp': 'osm_polygon',
    'osm_3.tif': 'osm_3m',
    'osm_075.tif': 'osm_075m',
    'microsoft.shp': 'microsoft_polygon',
    'gdal.shp': 'gdal_polygon',
    'gdal_microsoft3m.shp': 'gdal_polygon_microsoft3m'
}
_RESOURCE_MAP = {name:i for i, (_, name) in enumerate(_POSTFIX_MAP.items())}

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class PlanetBuildingResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://project.inria.fr/aerialimagelabeling/ and follow the instructions there.", **kwargs)


@register_dataset(NAME)
class PlanetBuildingPaired(Dataset):
    """
    - **homepage**: https://project.inria.fr/aerialimagelabeling/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "full",
        city_names = ['munich', 'berlin'],
        img_type = 'planet_SR',
        mask_type = 'microsoft_3m',
        gdal_poly_type = 'microsoft_polygon', # either osm or microsoft
        gt_poly_type = 'microsoft_polygon', # either osm or microsoft
        skip_integrity_check: bool = True,
        crop_size =  [256, 256],
        stride: int = [192, 192],
        mask_scale=1,
        apply_align = False
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
        self.city_names = city_names
        self.img_type = img_type
        self.mask_type = mask_type
        self.gdal_poly_type = gdal_poly_type
        self.gt_poly_type = gt_poly_type
        self.apply_align = apply_align
        self.mask_scale = mask_scale
        self.geo_transforms = {}
        # assert self.poly_type in ['microsoft_polygon', 'osm_polygon'], 'Invalid type of the shape file!'

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:

        city_resources = []
        for city in self.city_names:
            city_resource = PlanetBuildingResource(
                file_name = city,
                preprocess = None,
                sha256 = None
            )
            city_resources.append(city_resource)

        return city_resources

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
        flag = False
        for i, (key, value) in enumerate(_POSTFIX_MAP.items()):
            if path.name.endswith(key):
                flag = True
                return i

        # if not flag:
        #     pdb.set_trace()
        return len(_POSTFIX_MAP)

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



    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        dp = None
        for city_name, city_resource in zip(self.city_names, resource_dps):

            # img_dp, mask_dp, osm_dp, microsoft_dp, _ = Demultiplexer(city_resource, 5, self._classify_dp)
            resource_list = Demultiplexer(city_resource, len(_RESOURCE_MAP)+1, self._classify_dp)
            img_dp = resource_list[_RESOURCE_MAP[self.img_type]]
            mask_dp = resource_list[_RESOURCE_MAP[self.mask_type]]
            gdal_shp_dp = resource_list[_RESOURCE_MAP[self.gdal_poly_type]]
            gt_shp_dp = resource_list[_RESOURCE_MAP[self.gt_poly_type]]

            # shp_dp = eval(f'{self.poly_type}_dp')

            crop_size, stride = self.crop_size, self.stride

            clip_dir_name = 'clip_paired'
            clip_dir = os.path.join(self.root, '{}_c{}_s{}'.format(clip_dir_name, crop_size[0], stride[0]), city_name)
            clip_dir_img = os.path.join(self.root, '{}_c{}_s{}'.format(clip_dir_name, crop_size[0], stride[0]),
                                        city_name, f'{self.img_type}_dir')
            clip_dir_mask = os.path.join(self.root, '{}_c{}_s{}'.format(clip_dir_name, crop_size[0], stride[0]),
                                         city_name, f'{self.mask_type}_dir')

            os.makedirs(clip_dir, exist_ok=True)
            print(f'Clipping the dataset to patches of sizes {crop_size} x {crop_size}. May take a while...')
            img_paths = []
            mask_paths = []
            gdal_shp_paths = []
            gt_shp_paths = []

            for img_path, mask_path, gdal_shp_path, gt_shp_path in zip(img_dp, mask_dp, gdal_shp_dp, gt_shp_dp):
                img_path = img_path[0]
                mask_path = mask_path[0]
                gdal_shp_path = gdal_shp_path[0]
                gt_shp_path = gt_shp_path[0]

                img_paths.append(img_path)
                mask_paths.append(mask_path)
                gdal_shp_paths.append(gdal_shp_path)
                gt_shp_paths.append(gt_shp_path)

            assert len(img_paths) == 1, "please make sure there is only one set of files in each city"

            img_tif = rasterio.open(img_path)
            mask_tif = rasterio.open(mask_path)
            H, W = img_tif.shape
            # assert img_tif.shape == mask_tif.shape

            self.geo_transforms[city_name] = img_tif.transform

            # Get the boxes to crop the images, masks, or polygons
            crop_boxes = self.get_crop_boxes(H, W, crop_size, stride)
            crop_boxes_mask = self.get_crop_boxes(
                H * self.mask_scale,
                W * self.mask_scale,
                np.array(crop_size) * self.mask_scale,
                np.array(stride) * self.mask_scale
            )

            print(f'clipping the images for {city_name}...')
            # split images according to the boxes (if not exists)
            if not os.path.exists(clip_dir_img):
                os.makedirs(clip_dir_img)
                self.split_images(
                    image=img_tif.read(),
                    crop_boxes=crop_boxes,
                    out_dir=clip_dir_img,
                    img_name=img_path.split('/')[-1].split('.')[0],
                )

            print(f'clipping the masks for {city_name}...')
            # split masks
            if not os.path.exists(clip_dir_mask):
                os.makedirs(clip_dir_mask)
                self.split_images(
                    image=mask_tif.read(),
                    crop_boxes=crop_boxes_mask,
                    out_dir=clip_dir_mask,
                    img_name=mask_path.split('/')[-1].split('.')[0],
                )

            print(f'clipping the polygon for {city_name}...')
            gdal_ann_file_name = f'ann_{self.img_type}_{self.mask_type}_{self.gdal_poly_type}.json'
            gt_ann_file_name = f'ann_{self.img_type}_{self.mask_type}_{self.gt_poly_type}.json'
            ann_file_names = [gdal_ann_file_name, gt_ann_file_name]
            shp_paths = [gdal_shp_path, gt_shp_path]

            for ann_file_name, shp_path in zip(ann_file_names, shp_paths):
                ann_path = os.path.join(clip_dir, ann_file_name)
                if not os.path.exists(ann_path):
                    annotations = self.split_polygons(
                        shp_file=gpd.read_file(shp_path),
                        transform=img_tif.transform,
                        crop_boxes=crop_boxes,
                        # out_path=ann_path,
                        H=H, W=W
                    )

                    image_infos = self.get_image_infos(
                        crop_boxes, clip_dir,
                        img_path.split('/')[-1].split('.')[0],
                        mask_path.split('/')[-1].split('.')[0],
                        mask_scale=self.mask_scale
                    )

                    ann_dict = dict(
                        images=image_infos,
                        categories=[{"id": 1, "name": 'building'}],
                        annotations=annotations
                    )

                    with open(os.path.join(clip_dir, ann_file_name), 'w') as f:
                        f.write(json.dumps(ann_dict))

            # clip_dir = self._clip_city_images(city_name, [img_dp, mask_dp, shp_dp])
            if self.apply_align:
                aligned_ann_file_name = f'ann_{self.img_type}_{self.mask_type}_{self.gdal_poly_type}_aligned.json'
                if os.path.exists(aligned_ann_file_name):
                    self.coco = COCO(os.path.join(clip_dir, aligned_ann_file_name))
                else:
                    self.coco = COCO(os.path.join(clip_dir, ann_file_name))
                    aligned_ann = self.align_ann(self.coco)
                    with open(os.path.join(clip_dir, aligned_ann_file_name), 'w') as f:
                        f.write(json.dumps(aligned_ann))
                    self.coco = aligned_ann

            else:
                self.gdal_coco = COCO(os.path.join(clip_dir, gdal_ann_file_name))
                self.gt_coco = COCO(os.path.join(clip_dir, gt_ann_file_name))

            img_ids = self.gdal_coco.getImgIds()
            data_infos = []
            total_ann_ids = []
            for i in img_ids:
                gdal_ann_ids = self.gdal_coco.getAnnIds(imgIds=[i])
                gt_ann_ids = self.gt_coco.getAnnIds(imgIds=[i])

                if len(gdal_ann_ids) > 0:
                    gdal_info = self.gdal_coco.loadImgs([i])[0]
                    gt_info = self.gt_coco.loadImgs([i])[0]

                    gdal_info['filename'] = gdal_info['file_name']
                    gt_info['filename'] = gt_info['file_name']

                    data_infos.append(dict(gdal_info=gdal_info, gt_info=gt_info))
                    # total_ann_ids.extend(ann_ids)

            # assert len(set(total_ann_ids)) == len(
            #     total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"

            if dp is None:
                dp = Mapper(iter(data_infos), self._parse_ann_info)
            else:
                dp.concat(Mapper(iter(data_infos), self._parse_ann_info))

        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)

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
        base_len = {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'train_val': _TRAIN_LEN + _VAL_LEN,
            'test': _TEST_LEN

        }[self._split]
        if self.city_names is not None:
            base_len = int(base_len / 5 * len(self.city_names))

        return base_len
