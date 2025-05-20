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

from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    Zipper,
    ZipperLongest,
    Concater
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

NAME = "love_da"
_TRAIN_RURAL = 1366
_TRAIN_URBAN = 1156

_VAL_RURAL = 992
_VAL_URBAN = 677

_TEST_RURAL = 976
_TEST_URBAN = 820


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class LoveDAResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download Train.zip
        wget https://zenodo.org/record/5706578/files/Train.zip
        # Download Val.zip
        wget https://zenodo.org/record/5706578/files/Val.zip
        # Download Test.zip
        wget https://zenodo.org/record/5706578/files/Test.zip
        """
        super().__init__('Download the data into the root directory according to the download command in https://github.com/Junjue-Wang/LoveDA',
                         **kwargs)

@register_dataset(NAME)
class LoveDA(Dataset):
    """
    - **github link**: https://github.com/Junjue-Wang/LoveDA
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split = "train_rural",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        # assert split in ['train_rural', 'train_urban', 'val_rural', 'val_urban', 'test_rural', 'test_urban']
        self._split = split
        if type(self._split) == str:
            self._split = [self._split]

        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128]]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'Train.zip': '62e1672dfd24f4811cf6ecc1fd1c476eb140f2a351eec3d22de25cc4de83f4e9',
        'Val.zip': '91632a1e10d0014dda2b7805886a6bbfcbb9dac97c7da3db60a9abe4a5cab299',
        'Test.zip': '890e2f86800af626cae64eb8a03160c29a341aaa837cc81af74bb808013cd52e'
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:
        train_resource = LoveDAResource(
            file_name = 'Train.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['Train.zip']
        )

        val_resource = LoveDAResource(
            file_name = 'Val.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['Val.zip']
        )

        test_resource = LoveDAResource(
            file_name = 'Test.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['Test.zip']
        )

        return [train_resource, val_resource, test_resource]

    def _prepare_sample(self, data):

        if self._split[0].startswith('test'):
            image_data = data
            image_path, image_buffer = image_data
            img_info = {'filename':image_path,
                        'img_id': image_path.split('/')[-1].split('.')[0]}
        else:
            image_data, ann_data = data
            image_path, image_buffer = image_data
            ann_path, ann_buffer = ann_data

            img_info = {'filename':image_path,
                        'img_id': image_path.split('/')[-1].split('.')[0],
                        'ann':{'seg_map': ann_path}}

        return img_info

    def _classify_split(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('train.txt'):
            return 0
        elif path.name.endswith('val.txt'):
            return 1
        elif path.name.endswith('test.txt'):
            return 2

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.parent.name == 'images_png':
            if path.parents[1].name == 'Rural':
                return 0
            else:
                return 1
        elif path.parent.name == 'masks_png':
            if path.parents[1].name == 'Rural':
                return 2
            else:
                return 3
        else:
            return None

    def _split_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        return data[1].decode('UTF-8')

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name.split('.')[0]

    def _dp_key_fn(self, data):
        path = pathlib.Path(data[0][0])
        return path.name.split('.')[0]


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        train_rural_img, train_urban_img, train_rural_ann, train_urban_ann = Demultiplexer(
            resource_dps[0], 4, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        val_rural_img, val_urban_img, val_rural_ann, val_urban_ann = Demultiplexer(
            resource_dps[1], 4, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        test_rural_img, test_urban_img, test_rural_ann, test_urban_ann = Demultiplexer(
            resource_dps[2], 4, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )


        try:
            dps = []
            is_test_split = False
            for cur_split in self._split:
                if cur_split.startswith('test'):
                    is_test_split = True
                    cur_dp = eval(f'{cur_split}_img')
                else:
                    assert not is_test_split
                    cur_dp = Zipper(eval(f'{cur_split}_img'), eval(f'{cur_split}_ann'))
                dps.append(cur_dp)

            dp = Concater(*dps)

        except NameError:
            raise NameError('One of the the split names is invalid! It should be one of the following: \
                             train_rural | train_urban | val_rural | val_urban | test_rural | test_urban ')

        # dp = Zipper(img_dp, ann_dp)

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        length = 0
        for cur_split in self._split:
            length += eval(f'_{cur_split.upper()}')

        return length

if __name__ == '__main__':
    dp = Landslide4Sense('./')
