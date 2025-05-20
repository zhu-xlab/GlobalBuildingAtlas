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
    Concater,
    FileOpener
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

NAME = "dotav20"
_TRAIN_LEN = 5862
_VAL_LEN = 5863
_TRAIN_VAL_LEN = 5862 + 5863
_TEST_LEN = 11738


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class DOTAv20Resource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download SeasonNet data manually from https://captain-whu.github.io/DOTA/dataset.html:
        """
        super().__init__('For data download, please refer to \
                         https://captain-whu.github.io/DOTA/dataset.html. \
                         Put the downloaded folders from DOTAv1.0 under /${root}/DOTA-v1.0, \
                         and the folders from DOTAv2.0 under /${root}/DOTA-v2.0. \
                         Here ${root} is the root directory given as the argument.',
                         **kwargs)




@register_dataset(NAME)
class DOTAv20(Dataset):
    """
    - **home page**: https://captain-whu.github.io/DOTA/dataset.html
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split == 'train' or split == 'val' or split == 'test' or split == 'trainval'
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'DOTA-v1.0/train/images/part1.zip': '8f2a06820798c3be88bbeef31a7b43e54ebe84adfb42c39085a87d9dea3514b7',
        'DOTA-v1.0/train/images/part2.zip': '18dd413b1d7ce9e1b5cfbbf18e524ebcc5be5064da7a4a8768a38b493a6c2905',
        'DOTA-v1.0/train/images/part3.zip': '5ff5ef2d492bbd55139cf3f39f839adeaa0fc8dfee0174dd7cb148402da0c823',
        'DOTA-v2.0/train/images/part4.zip': '6b9444666d7a238b78bea59170a3d7e7a3a2fbbdd924ff051df21ef233775ba5',
        'DOTA-v2.0/train/images/part5.zip': '625e0fe852fe47e103a2f669c80b25fbb1e0a3bac87cee11b0aace05a5223cc1',
        'DOTA-v2.0/train/images/part6.zip': '499557d9533e15a0b90a2066219fe699dd9a1f29fb354aaaaab968da042fcc4c',
        'DOTA-v1.0/val/images/part1.zip': 'e8f5602dc22c5142b3d2f26982c54df4e9e5e9b6b92015bc20a434e12b45ca9d',
        'DOTA-v2.0/val/images/part2.zip': 'db3a5d23d2fe4cfe2b936ac4db3d55611cdc7ffa82e120d6b15deaf1de91fbdc',
        'DOTA-v1.0/test/images/part1.zip': 'e8f5602dc22c5142b3d2f26982c54df4e9e5e9b6b92015bc20a434e12b45ca9d',
        'DOTA-v1.0/test/images/part2.zip': 'ef54c354fcfe9bc9aba53d7b8e6f7009304e5b9ea87d35af482d9c49438b2356',
        'DOTA-v2.0/test-dev/part3.zip': 'e5f66fad2d899364c1b9cf647091f22b179c79aa89627a72b6c3c2b158ef72c4',
        'DOTA-v2.0/test-dev/part4.zip': '2242c99af2ad4551fdaa8bb2036d03234ed869c0beb46d25dd4d7586bbaddbfa',
        'DOTA-v2.0/test-dev/part5.zip': '1344f6d0ad9099fe90536864883d29bd0f74c98ded7704990592160e7e8baec6',
        'DOTA-v2.0/test-dev/part6.zip': '20d5cf63b9e9d2be3bbc57210fdd467e91008791bad0646f479c4368afdb948c',
        'DOTA-v2.0/test-dev/part7.zip': '0c025fe29a79f62d79ed8f0f525db19c0f784683f092e6adf11b064823cc09a0',
        'DOTA-v2.0/test-dev/part8.zip': '73a9c681630800a51679965b74704234c084d72f0784163548d9029de73b2a36',
        'DOTA-v2.0/test-dev/part9.zip': 'abe8bdfb5cfb4e0fe6799551e50698a4df5a4b7a3ee189aeea12d0a7987470c4',
        'DOTA-v2.0/test-dev/part10.zip': 'b476407888b16e5358ee580e6dfd785b551912ea2ee0549b9f078520294c2fe4',
        'DOTA-v2.0/train/labelTxt-v2.0/DOTA-v2.0_train.zip': '44fe9f9539835bde99178e76ccf90a04f4a26218554ce612d16a42cb429a1942',
        'DOTA-v2.0/train/labelTxt-v2.0/DOTA-v2.0_train_hbb.zip': '0ebf6d3216524837f459d6ba762095f6eabee404c7ddfba959878dd493d1a7d7',
        'DOTA-v2.0/val/labelTxt-v2.0/DOTA-v2.0_train.zip': '44fe9f9539835bde99178e76ccf90a04f4a26218554ce612d16a42cb429a1942',
        'DOTA-v2.0/val/labelTxt-v2.0/DOTA-v2.0_train_hbb.zip': '3ab5a47bafa77f5e8fa7f8db60826753614bdecb19f1eae2bdcf2edc4dc46875',
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:
        resources = []
        for path, checksum in self._CHECKSUMS.items():
            resource = DOTAv20Resource(
                file_name = path,
                preprocess = 'extract',
                sha256 = checksum
            )
            resources.append(resource)

        return resources

    def _prepare_sample(self, data):

        image_data, ann_data = data[1]
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img_info = {'filename':image_path,
                    'img_id': image_path.split('/')[-1].split('.')[0],
                    'ann':{'ann_path': ann_path}}

        return img_info

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        train_img_dp = Concater(*resource_dps[:6])
        train_ann_dp = resource_dps[18]
        train_ann_hbb_dp = resource_dps[19]

        val_img_dp = Concater(*resource_dps[6:8])
        val_ann_dp = resource_dps[20]
        val_ann_hbb_dp = resource_dps[21]

        test_img_dp = Concater(*resource_dps[8:18])

        img_dp = eval(f'{self._split}_img_dp')
        ann_dp = eval(f'{self._split}_ann_dp') if self._split is not 'test' else None
        ann_hbb_dp = eval(f'{self._split}_ann_hbb_dp') if self._split is not 'test' else None

        pdb.set_trace()

        dp = Zipper(img_dp, ann_dp, ann_hbb_dp)

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'test': _TEST_LEN,
            'trainval': _TRAIN_VAL_LEN
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
