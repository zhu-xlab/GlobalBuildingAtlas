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

NAME = "hyper_11k"
_TRAIN_LEN = 11483
_TRAIN_10K_LEN = 10000
_TRAIN_1K_LEN = 1000

@register_info(NAME)
def _info() -> Dict[str, Any]:
    # return dict(categories=read_categories_file(NAME))
    return dict(categories=None)

class Hyper11kResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download Hyper11k data manually:
        """
        super().__init__('For data download, please refer to https://hyspecnet.rsim.berlin/',
                         **kwargs)

"""
Currently only support uses for self-supervised learning. Annotations are not loaded.
"""
@register_dataset(NAME)
class Hyper11k(Dataset):

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

        img_resource = Hyper11kResource(
            file_name='hyspecnet-11k/patches',
            # preprocess='extract',
        )

        return img_resource,

    def _prepare_sample(self, data):

        image_path, image_buffer = data

        img_info = {
            'dataset_name': 'hyper_11k',
            'filename':image_path,
            'img_id': image_path.split('/')[-1].split('.')[0],
            'latitude': None,
            'longitude': None,
            'resolution': None # manually set to 0.6 resolution
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

    def _is_spectral(self, data):
        return 'SPECTRAL_IMAGE' in data[0]

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        train_img_dp, = resource_dps
        train_img_dp = train_img_dp.filter(filter_fn=self._is_spectral)
        train_1k_img_dp = itertools.islice(train_img_dp, 1000)
        train_10k_img_dp = itertools.islice(train_img_dp, 10000)
        img_dp = eval(f'{self._split}_img_dp')

        ndp = Mapper(img_dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'train_1k': _TRAIN_1K_LEN,
            'train_10k': _TRAIN_10K_LEN,
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
