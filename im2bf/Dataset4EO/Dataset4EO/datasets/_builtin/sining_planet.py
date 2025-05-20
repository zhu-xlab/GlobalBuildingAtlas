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

NAME = "sining_planet"
_TRAIN_LEN = 93022
_VAL_LEN = 23288
_TEST_LEN = 11415
_TEST_1k_LEN = 1000

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class SiningPlanetResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download CrowdAI data manually:
        """
        super().__init__('For data download, please go to https://www.aicrowd.com/challenges/mapping-challenge/dataset_files',
                         **kwargs)

@register_dataset(NAME)
class SiningPlanetDataset(Dataset):
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

        assert split in ['train', 'val', 'test', 'test_1k']
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info
        self.cat_ids = [100]
        self.cat2label = {100: 1}
        self.CLASSES = ('building', 'background')
        self.PALETTE = [[128, 0, 0], [0, 128, 0]]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:

        img_resource = SiningPlanetResource(
            file_name = 'bf_test_data/image',
            preprocess = None,
        )

        gt_resource = SiningPlanetResource(
            file_name = 'bf_test_data/mask',
            preprocess = None,
        )

        return [img_resource, gt_resource]

    def _parse_dp(self, data):
        img_path = data[0][0]
        gt_path = data[1][0]

        result = dict(
            img_path=img_path,
            seg_path=gt_path
        )

        return result


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        img_dp, gt_dp = resource_dps
        dp = Zipper(img_dp, gt_dp)
        dp = Mapper(dp, self._parse_dp)

        return dp

    def __len__(self) -> int:
        return 100

if __name__ == '__main__':
    dp = Landslide4Sense('./')
