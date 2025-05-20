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
from itertools import chain
import glob


from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    Zipper,
    IterableWrapper,
    Concater
)
# from torchdata.datapipes.iter import Concater

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

NAME = "global_building_raster"

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class GlobalBuildingResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://project.inria.fr/aerialimagelabeling/ and follow the instructions there.", **kwargs)


@register_dataset(NAME)
class GlobalBuildingRaster(Dataset):
    """
    """

    def __init__(
        self,
        root,
        in_base_dirs,
        out_root,
        *,
        split: str = "full",
        skip_integrity_check: bool = True,
        crop_size =  [256, 256],
        stride: int = [192, 192],
        patchify = False,
        filter_post_fix=None,
        data_type='sr'
    ) -> None:

        assert split in ('train', 'val', 'test', 'full')

        self._split = split
        self.root = root
        if type(in_base_dirs) == str:
            in_base_dirs = [in_base_dirs]
        self.in_base_dirs = in_base_dirs
        self.out_root = out_root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[0,0,0], [255,255,255]]
        self.crop_size = crop_size
        self.stride=stride
        self.cat_ids = [1]
        self.cat2label = {1: 1}
        self.patchify = patchify
        self.filter_post_fix = filter_post_fix
        assert data_type in ['sr', 'bf']
        self.data_type = data_type
        # assert self.poly_type in ['microsoft_polygon', 'osm_polygon'], 'Invalid type of the shape file!'

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:

        resources = []

        for in_base_dir in self.in_base_dirs:
            cur_resource = GlobalBuildingResource(
                file_name = in_base_dir,
                preprocess = None,
                sha256 = None
            )
            resources.append(cur_resource)

        return resources

    def _classify_dp(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('.tif'):
            return 0
        return 1

    def _filter_dp_started_flag(self, data):
        path = pathlib.Path(data[0])
        # if 'GUF04_DLR_v02_e005_n50_e010_n45_OGR04' in str(path):
        # if 'GUF04_DLR_v02_w080_n10_w075_n05_OGR04' in str(path):
        #     return True
        # else:
        #     return False

        parent = path.parent
        rel_parent = os.path.relpath(parent, self.root)

        if os.path.exists(os.path.join(self.out_root, rel_parent, 'started.txt')):
            return False

        return True

    def _filter_dp_finished_flag(self, data):
        path = pathlib.Path(data[0])
        parent = path.parent
        rel_parent = os.path.relpath(parent, self.root)

        if os.path.exists(os.path.join(self.out_root, rel_parent, 'finished.txt')):
            return False

        return True

    def _filter_dp_tif(self, data):
        path = pathlib.Path(data[0])
        return path.endswith('.tif')

    def _filter_dp_folder_exist(self, data):
        path = pathlib.Path(data[0])
        if self.patchify:
            crop_boxes = [str(x) for x in data[1]]
            path = str(path).split('.tif')[0]
            path = os.path.join(path, '_'.join(crop_boxes)) + '.tif'

        rel_path = os.path.relpath(path, self.root).split('.')[0]
        out_path = os.path.join(self.out_root, rel_path)
        return not os.path.exists(out_path)

    def _filter_by_post_fix(self, data):
        path = str(pathlib.Path(data[0]))
        return self.filter_post_fix is None or path.endswith(self.filter_post_fix)

    def calculate_patches(self, H, W, w):
        """
        Calculate the sizes and starting coordinates of patches to cover an image.
        
        :param H: Height of the image.
        :param W: Width of the image.
        :param w: Desired rough size of the square patches.
        :return: List of tuples, each representing a patch in the form (start_y, start_x, height, width)
        """
        # Calculate the number of patches in each dimension (rounding up to cover the whole image)
        num_patches_height = -(-H // w)  # Equivalent to ceil(H / w) for integers
        num_patches_width = -(-W // w)  # Equivalent to ceil(W / w) for integers
        
        # Adjust the size of each patch to evenly cover the image
        patch_height = H / num_patches_height
        patch_width = W / num_patches_width
        
        patches = []
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                start_y = int(i * patch_height)
                start_x = int(j * patch_width)
                # Ensure the patches at the edges do not exceed the image dimensions
                end_y = int(start_y + patch_height) if i < num_patches_height - 1 else H
                end_x = int(start_x + patch_width) if j < num_patches_width - 1 else W
                cur_height = end_y - start_y
                cur_width = end_x - start_x
                patches.append((start_x, start_y, cur_height, cur_width))
        
        return patches


    def _parse_dp(self, data):
        path = pathlib.Path(data[0])
        if self.patchify:
            H, W = rasterio.open(path).shape
            crop_boxes = self.calculate_patches(H, W, self.crop_size[0])
            return [(path, x) for x in crop_boxes]


        return path

    def _datapipe(self, resource_dps):
        resource_dp = Concater(*resource_dps)
        raster_dp = resource_dp.filter(self._filter_dp_started_flag).filter(self._filter_dp_finished_flag)
        # raster_dp = raster_dp.filter(self._filter_dp_folder_exist)
        raster_dp = raster_dp.filter(self._filter_by_post_fix)

        if self.patchify:
            temp = Mapper(raster_dp, self._parse_dp)
            raster_dp = chain.from_iterable(temp)
            raster_dp = IterableWrapper(raster_dp)
        else:
            raster_dp = Mapper(raster_dp, self._parse_dp)

        # raster_dp = raster_dp.filter(self._filter_dp_folder_exist)
        raster_dp = raster_dp.filter(self._filter_dp_folder_exist)

        return raster_dp

    def __len__(self) -> int:

        return 195104 if self.data_type == 'sr' else 960
        # return len(list(self.raster_dp))

