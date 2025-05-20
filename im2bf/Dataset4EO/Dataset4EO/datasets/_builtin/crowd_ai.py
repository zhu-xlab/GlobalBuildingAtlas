import os
import tarfile
import enum
import functools
import itertools
import pathlib
from tqdm import tqdm
import h5py
import torch
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union
from xml.etree import ElementTree
from Dataset4EO import transforms
import pdb
import numpy as np
from pycocotools.coco import COCO as COCO
import geojson
import shapely

from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    Zipper,
    Concater,
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

NAME = "crowd_ai"
_TRAIN_LEN = 280741
_VAL_LEN = 60317
_TRAIN_SMALL_LEN = 8366
_VAL_SMALL_LEN = 1820

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class CrowdAIResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download CrowdAI data manually:
        """
        super().__init__('For data download, please go to https://www.aicrowd.com/challenges/mapping-challenge/dataset_files',
                         **kwargs)

@register_dataset(NAME)
class CrowdAIDataset(Dataset):
    """
    """
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split in ['train', 'val', 'train_small', 'val_small']
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info
        self.cat_ids = [100]
        self.cat2label = {100: 1}
        self.CLASSES = ('background', 'landslide')
        self.PALETTE = [[128, 0, 0], [0, 128, 0]]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'train': '459b2ef9c4ab0bd24a03cb6e4cb970857106df63bfd7c8356a83f818e045942c',
        'val': 'd88ad35ef638231977883a168bc9dadf7eb67234358b59eb83d21cd7999309e4'
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:

        train_resource = CrowdAIResource(
            file_name = '8e089a94-555c-4d7b-8f2f-4d733aebb058_train.tar',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['train']
        )

        val_resource = CrowdAIResource(
            file_name = '0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val.tar',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['val']
        )

        return [train_resource, val_resource]


    def _prepare_sample(self, img_id):

        def fix_polygons(polygons, buffer=0.0):
            polygons_geom = shapely.ops.unary_union(polygons)  # Fix overlapping polygons
            polygons_geom = polygons_geom.buffer(buffer)  # Fix self-intersecting polygons and other things
            fixed_polygons = []
            if polygons_geom.geom_type == "MultiPolygon":
                for poly in polygons_geom.geoms:
                    fixed_polygons.append(poly)
            elif polygons_geom.geom_type == "Polygon":
                fixed_polygons.append(polygons_geom)
            else:
                raise TypeError(f"Geom type {polygons_geom.geom_type} not recognized.")
            return fixed_polygons

        img_info = self.coco.loadImgs([img_id])[0]
        res_dict = dict()

        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
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
            features.append(ann['segmentation'])
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if len(gt_bboxes) > 0:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if len(gt_bboxes_ignore) > 0:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        """
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            features=features
        )

        res = dict(
            filename=os.path.join(self.img_base_dir, img_info['file_name']),
            # maskname=img_info['mask_name'],
            img_id=img_id,
            ann=ann,
            height=img_info['height'],
            width=img_info['width']
        )
        """

        new_features = []
        for feature in features:
            if len(feature) > 1:
                pdb.set_trace()
            for x in feature:
                if len(np.array(x).shape) > 2:
                    pdb.set_trace()
            feature = [np.array(x).reshape(-1,2).tolist() for x in feature]
            exterior = feature[0]
            interiors = [] if len(feature) == 1 else feature[1:]
            polygon = geojson.Polygon([exterior], interiors)
            # polygon_shape = shapely.geometry.shape(polygon)
            # if not polygon_shape.is_valid:
            #     fixed_polygon_shape = fix_polygons(polygon_shape, 0.0001)

            new_features.append(polygon)

        res = dict(
            img_id=img_id,
            img_path = os.path.join(self.img_base_dir, img_info['file_name']),
            features=new_features
        )

        return res

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        train_ann_path = os.path.join(self.root, '8e089a94-555c-4d7b-8f2f-4d733aebb058_train/train/annotation.json')
        train_small_ann_path = os.path.join(self.root, '8e089a94-555c-4d7b-8f2f-4d733aebb058_train/train/annotation-small.json')
        val_ann_path = os.path.join(self.root, '0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation.json')
        val_small_ann_path = os.path.join(self.root, '0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val/val/annotation-small.json')

        ann_path = eval(f'{self._split}_ann_path')
        self.img_base_dir = os.path.join('/'.join(ann_path.split('/')[:-1]), 'images')

        coco = COCO(ann_path)
        self.coco = coco
        img_ids = self.coco.getImgIds()
        dp = Mapper(img_ids, self._prepare_sample)

        return dp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'train_small': _TRAIN_SMALL_LEN,
            'val_small': _VAL_SMALL_LEN,
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
