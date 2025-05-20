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

NAME = "nwpu_vhr10_v2"
_TRAIN_LEN = 5862
_VAL_LEN = 5863
_TRAIN_VAL_LEN = 5862 + 5863
_TEST_LEN = 11738
_TEST_1K_LEN = 1000

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class NWPU_VHR10_V2Resource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download nwpu_vhr10_v2 data manually:
        """
        super().__init__('For data download, please refer to https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC',
                         **kwargs)

@register_dataset(NAME)
class DIOR(Dataset):
    """
    - **paper link**: https://arxiv.org/abs/1909.00133?context=cs.LG.html
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split in ['train', 'val', 'test', 'test_1k', 'trainval', 'test_10']
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'ImageSets.zip': '682a3e858d9c76fa7727031ddd1a0619e9cb85a1aee354265895e1856df7742c',
        'Annotations.zip': 'e5ae9ba732cf2bc1c944de7a5caf5631929488cb10cb797b0b3722da2b0c6d72',
        'JPEGImages-trainval.zip': '5e3757944739cffc8ba7de537db868e0a0ad86c8dfb44fa42c9dc88d6f327747',
        'JPEGImages-test.zip': '8aa1e0e1496fd7a9f8cec7018ff3c9a196e9f9f47ef2aba2cb36f2dbf1368375'
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:

        split_resource = DIORResource(
            file_name = 'ImageSets.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['ImageSets.zip']
        )

        ann_resource = DIORResource(
            file_name = 'Annotations.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['Annotations.zip']
        )

        img_trainval_resource = DIORResource(
            file_name = 'JPEGImages-trainval.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['JPEGImages-trainval.zip']
        )

        img_test_resource = DIORResource(
            file_name = 'JPEGImages-test.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['JPEGImages-test.zip']
        )

        return [split_resource, ann_resource, img_trainval_resource, img_test_resource]

    def _prepare_sample(self, data):

        image_data, ann_data = data[1]
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img_info = {'filename':image_path,
                    'img_id': image_path.split('/')[-1].split('.')[0],
                    'ann':{'ann_path': ann_path}}

        return img_info

    def _classify_split(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('train.txt'):
            return 0
        elif path.name.endswith('val.txt'):
            return 1
        elif path.name.endswith('test.txt'):
            return 2
        elif path.name.endswith('test_1k.txt'):
            return 3
        elif path.name.endswith('test_10.txt'):
            return 4
        else:
            raise ValueError(f'name {path.name} not found as a split file')

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('.txt'):
            return 0
        elif path.name.endswith('jpg'):
            return 1
        elif path.name.endswith('xml') and path.parent.name == 'Horizontal Bounding Boxes':
            return 2
        elif path.name.endswith('xml') and path.parent.name == 'Oriented Bounding Boxes':
            return 3
        else:
            return None

    def _classify_ann(self, data):
        path = pathlib.Path(data[0])
        if path.name.endswith('xml') and path.parent.name == 'Horizontal Bounding Boxes':
            return 0
        elif path.name.endswith('xml') and path.parent.name == 'Oriented Bounding Boxes':
            return 1
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

        split_dp, ann_dp, trainval_img_dp, test_img_dp = resource_dps

        """ prepare split """
        train_split, val_split, test_split, test_split_1k, test_split_10 = Demultiplexer(split_dp, 5, self._classify_split)
        train_split = LineReader(train_split)
        val_split = LineReader(val_split)
        test_split = LineReader(test_split)
        test_1k_split = LineReader(test_split_1k)
        test_10_split = LineReader(test_split_10)

        if self._split == 'trainval':
            split_dp = train_split.concat(val_split)
        else:
            split_dp = eval(f'{self._split}_split')

        """ prepare images """
        img_dp = Concater(trainval_img_dp, test_img_dp)

        """ prepare annotations """
        ann_dp_h, ann_dp_o = Demultiplexer(
            ann_dp, 2, self._classify_ann, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        """ correlate the images and the horizontal annotations"""
        img_ann_dp = IterKeyZipper(
            img_dp, ann_dp_h,
            key_fn=self._images_key_fn,
            ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=False
        )

        """ correlate the images, annotations and the split """
        dp = IterKeyZipper(
            split_dp, img_ann_dp,
            key_fn=self._split_key_fn,
            ref_key_fn=self._dp_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=False
        )
        # dp = Zipper(img_dp, ann_dp)

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'test': _TEST_LEN,
            'trainval': _TRAIN_VAL_LEN,
            'test_1k': _TEST_1K_LEN
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
