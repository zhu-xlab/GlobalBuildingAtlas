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
import math

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

NAME = "satlas_naip"
_TRAIN_LEN = 3206840
_TRAIN_10K_LEN = 10000
_TRAIN_1K_LEN = 1000

@register_info(NAME)
def _info() -> Dict[str, Any]:
    # return dict(categories=read_categories_file(NAME))
    return dict(categories=None)

class SatlasNAIPResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download SatlasNAIP data manually:
        """
        super().__init__('For data download, please refer to https://github.com/allenai/satlas/blob/main/SatlasPretrain.md',
                         **kwargs)

def mercator_to_geo(p, zoom=13, pixels=512):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)

"""
Currently only support uses for self-supervised learning. Annotations are not loaded.
"""
@register_dataset(NAME)
class SatlasNAIP(Dataset):

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split in ['train', 'train_1k', 'train_10k'] # only support for self-supervised learning currently
        self._split = split
        self.root = root
        # self._categories = _info()["categories"]
        self._categories = None
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def get_classes(self):
        return None
        # return self._categories

    def _resources(self) -> List[OnlineResource]:

        img_resource = SatlasNAIPResource(
            file_name='naip',
            # preprocess='extract',
        )

        return img_resource,

    def _prepare_sample(self, data):

        image_path, image_buffer = data

        mecators = image_path.split('/')[-1].split('.')[0]
        p = [int(x) for x in mecators.split('_')]
        long, lat = mercator_to_geo(p, zoom=17, pixels=1)

        img_info = {
            'dataset_name': 'satlas_naip',
            'filename':image_path,
            'img_id': mecators,
            'latitude': lat,
            'longitude': long,
            'resolution': 0.6 # manually set to 0.6 resolution
        }

        return img_info

    def _prepare_sample_no_ann(self, data):

        image_path, image_buffer = data
        img_info = {'filename':image_path,
                    'img_id': image_path.split('/')[-1].split('.')[0],
                    'ann': None}

        return img_info

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        dp, = resource_dps
        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        train_img_dp = ndp
        train_1k_img_dp = itertools.islice(train_img_dp, 1000)
        train_10k_img_dp = itertools.islice(train_img_dp, 10000)

        img_dp = eval(f'{self._split}_img_dp')


        return img_dp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'train_1k': _TRAIN_1K_LEN,
            'train_10k': _TRAIN_10K_LEN,
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
