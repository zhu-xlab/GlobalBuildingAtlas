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

NAME = "fair1m"
_TRAIN_LEN = 16488
_VAL_LEN = 8287
_TEST_LEN = 18021
_TEST_1K_LEN = 1000

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class Fair1MResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download FAIR1M data manually:
        """
        super().__init__('For data download, please refer to https://gaofen-challenge.com/benchmark',
                         **kwargs)

@register_dataset(NAME)
class Fair1M(Dataset):

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
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'train/part1/images.zip': 'e3ca90b4421a9521f9509f297062a16dea23458d37eb9862fdc24bc73b7d839d',
        'train/part2/images.zip': '4ddad9f787d0be738f0ea8e713aa7512490691611273e8dcc8f24946eff008c4',
        'validation/images.zip': '84e844bcc23488bf7b4b755f7f88d7398d371050400318820d6125309bf2932d',
        'test/images0.zip': 'dad6af498b9ab56fd28cefc39a058952f19a4ff19862a4ac90dd0defe3e76269',
        'test/images1.zip': '170bafe35644d1815ddc896d1c7a7a645e0a087b7c5f7891927aa1b32b55913d',
        'test/images2.zip': 'f5a959b47078b8fb22097cb96f2d95e00ff5415ea29f8296ea1e1b174e7c4003',
        'train/part1/labelXmls.zip': '259d15a329cd00a424159e01605b84322d4fc37207f1c6bb419e6f26b423f0ec',
        'train/part2/labelXmls.zip': '3bb2c597250a3f5c81c49a939b45e2308ecbf890910c5c7ebbfaefce171cb080',
        'validation/labelXmls.zip': '080c451545e285cd94fc62d02b0a7db182fd73be9c03205414dd178f03440901',
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:

        resources = []
        for path, checksum in self._CHECKSUMS.items():
            resource = Fair1MResource(
                file_name=path,
                preprocess='extract',
                sha256=checksum
            )
            resources.append(resource)

        return resources

    def _prepare_sample(self, data):

        image_data, ann_data = data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img_info = {'filename':image_path,
                    'img_id': image_path.split('/')[-1].split('.')[0],
                    'ann':{'ann_path': ann_path}}

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

        train_img_dp1, train_img_dp2, val_img_dp, test_img_dp1, \
                test_img_dp2, test_img_dp3, train_ann_dp1, train_ann_dp2, val_ann_dp = resource_dps
        train_img_dp = Concater(train_img_dp1, train_img_dp2)
        test_img_dp = Concater(test_img_dp1, test_img_dp2, test_img_dp3)
        train_ann_dp = Concater(train_ann_dp1, train_ann_dp2)

        img_dp = eval(f'{self._split}_img_dp')
        ann_dp = eval(f'{self._split}_ann_dp') if self._split != 'test' else None

        if self._split != 'test':
            img_ann_dp = IterKeyZipper(
                img_dp, ann_dp,
                key_fn=self._images_key_fn,
                ref_key_fn=self._anns_key_fn,
                buffer_size=INFINITE_BUFFER_SIZE,
                keep_key=False
            )
            ndp = Mapper(img_ann_dp, self._prepare_sample)
        else:
            img_ann_dp = img_dp
            ndp = Mapper(img_ann_dp, self._prepare_sample_no_ann)



        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'test': _TEST_LEN,
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
