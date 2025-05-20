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

NAME = "landslide4sense"
_TRAIN_LEN = 3799
_VAL_LEN = 245
_TEST_LEN = 800


class Landslide4SenseResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download SeasonNet data manually on:
        https://www.iarai.ac.at/landslide4sense/challenge/
        """
        super().__init__('For data download, please refer to https://www.iarai.ac.at/landslide4sense/challenge/',
                         **kwargs)
@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class Landslide4Sense(Dataset):
    """
    - **homepage**: https://www.iarai.ac.at/landslide4sense/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split in ['train', 'val', 'test']
        self._split = split

        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = ('background', 'landslide')
        self.PALETTE = [[128, 0, 0], [0, 128, 0]]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'TrainData.zip': '65538d45c153c989fe8869233061cc676019bd3a5a81978ccae9372142bc544d',
        'ValidData.zip': 'e85f9604e583a1023c2a49288a3239285e8bda24ab89cfa082ac5fdd9a21227e',
        'TestData.zip': '9506aa60e8754f95a9009fe395e236f80e48d37cba46589d5edc19b75663b6ab'
    }

    def _resources(self) -> List[OnlineResource]:
        train_archive = Landslide4SenseResource(
            file_name='TrainData.zip',
            preprocess='extract',
            sha256=self._CHECKSUMS['TrainData.zip']
        )
        val_archive = Landslide4SenseResource(
            file_name='ValidData.zip',
            preprocess='extract',
            sha256=self._CHECKSUMS['ValidData.zip']
        )
        test_archive = Landslide4SenseResource(
            file_name='TestData.zip',
            preprocess='extract',
            sha256=self._CHECKSUMS['TestData.zip']
        )

        return [train_archive, val_archive, test_archive]

    def _prepare_sample_dp(self, data):
        img_info = {}
        if self._split == 'train':
            (img_path, img_buffer), (ann_path, ann_buffer) = data
            img_info['ann'] = dict(seg_map=ann_path)
        else:
            (img_path, img_buffer) = data
            ann_path = 'None'

        img_info['filename'] = img_path
        return img_info

    def _prepare_sample(self, data):

        results = {}
        img_info = {}
        if self._split == 'train':
            (img_path, img_buffer), (ann_path, ann_buffer) = data
            ann = h5py.File(ann_path, 'r')['mask'][()]
            results['ann'] = ann
            img_info['ann'] = dict(seg_map=ann_path)
        else:
            (img_path, img_buffer) = data
            ann_path = 'None'

        img = h5py.File(img_path, 'r')['img'][()]
        img = torch.tensor(img).permute(2, 0, 1)
        img_info['filename'] = img_path


        results['img'] = img
        results['img_info'] = img_info

        return results

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.parent.name == 'img':
            return 0
        elif path.parent.name == 'mask':
            return 1

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        train_dp, val_dp, test_dp = resource_dps
        dp = eval(f'{self._split}_dp')

        img_dp, ann_dp = Demultiplexer(
            dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        dp = Zipper(img_dp, ann_dp) if self._split == 'train' else img_dp

        if not self.data_info:
            ndp = Mapper(dp, self._prepare_sample)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)
            tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomResizedCrop((128, 128), scale=[0.5, 1]))
            ndp = ndp.map(tfs)
        else:
            ndp = Mapper(dp, self._prepare_sample_dp)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'test': _TEST_LEN
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
