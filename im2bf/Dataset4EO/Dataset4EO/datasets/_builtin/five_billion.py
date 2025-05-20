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
import re

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

NAME = "five_billion"
_TRAIN_LEN = 117450
_TRAIN_10K_LEN = 10000
_TRAIN_1K_LEN = 1000

@register_info(NAME)
def _info() -> Dict[str, Any]:
    # return dict(categories=read_categories_file(NAME))
    return dict(categories=None)

class FiveBillionResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download FiveBillion data manually:
        """
        super().__init__('For data download, please refer https://x-ytong.github.io/project/Five-Billion-Pixels.html',
                         **kwargs)

def extract_lat_long_offsets(file_path):
    # Initialize latOffset and longOffset to None
    lat_offset = None
    long_offset = None

    # Regular expressions to match latOffset and longOffset
    lat_pattern = r"latOffset =\s*([\d\.-]+);"
    long_pattern = r"longOffset =\s*([\d\.-]+);"

    try:
        # Open the .rpb file and read its content
        with open(file_path, 'r') as file:
            content = file.read()

        # Use regular expressions to find latOffset and longOffset
        lat_match = re.search(lat_pattern, content)
        long_match = re.search(long_pattern, content)

        # Extract the values
        if lat_match:
            lat_offset = float(lat_match.group(1))
        if long_match:
            long_offset = float(long_match.group(1))

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    assert lat_offset is not None
    assert long_offset is not None

    return lat_offset, long_offset

"""
Currently only support uses for self-supervised learning. Annotations are not loaded.
"""
@register_dataset(NAME)
class FiveBillion(Dataset):

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

        img_resource = FiveBillionResource(
            file_name='images_splited_512_256',
            preprocess=None,
        )
        meta_data_resource = FiveBillionResource(
            file_name='meta_data_splited_512_256',
            preprocess=None
        )
        # meta_data_resource = FiveBillionResource(
        #     file_name='Coordinate_files.zip',
        #     preprocess='extract',
        #     sha256='7e5c197f424096fb27af6b3d0162126940f7f6106b8b8965e58df73979c3ee76'
        # )

        return [img_resource, meta_data_resource]

    def _prepare_sample(self, data):

        image_data, meta_data = data
        image_path, image_buffer = image_data
        meta_data_path, meta_data_buffer = meta_data
        lat_offset, long_offset = extract_lat_long_offsets(meta_data_path)

        img_info = {
            'dataset_name': 'five_billion',
            'filename':image_path,
            'img_id': image_path.split('/')[-1].split('.')[0],
            'latitude': float(lat_offset),
            'longitude': float(long_offset),
            'resolution': 4.0 # manually set to 4m resolution
        }

        return img_info

    def _meta_data_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        # return path.name.split('.')[0]
        return path.name.split('_patch')[0]

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('_patch')[0]

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        img_dp, meta_data_dp = resource_dps
        train_img_meta_data_dp = IterKeyZipper(
            img_dp, meta_data_dp,
            key_fn=self._images_key_fn,
            ref_key_fn=self._meta_data_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=False
        )
        train_1k_img_meta_data_dp = itertools.islice(train_img_meta_data_dp, 1000)
        train_10k_img_meta_data_dp = itertools.islice(train_img_meta_data_dp, 10000)

        img_meta_data_dp = eval(f'{self._split}_img_meta_data_dp')

        ndp = Mapper(img_meta_data_dp, self._prepare_sample)

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
