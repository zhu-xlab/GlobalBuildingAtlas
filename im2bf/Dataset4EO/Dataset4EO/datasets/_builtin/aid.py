import os
import tarfile
import enum
import functools
import pathlib
import glob
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
)

from torchdata.datapipes.map import SequenceWrapper

from Dataset4EO.datasets.utils import OnlineResource, HttpResource, Dataset
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

NAME = "aid"
_NUM_FILES = 4
_NUM_IMAGES = 10000


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


@register_dataset(NAME)
class AID(Dataset):
    """
    - **homepage**: https://captain-whu.github.io/AID/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        # There is currently no test split available
        assert split != 'test'

        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))
        self.root = root
        self.decom_dir = os.path.join(self.root, 'AID')
        self._categories = _info()["categories"]
        # self.CLASSES = ('background', 'landslide')
        # self.PALETTE = [[128, 0, 0], [0, 128, 0]]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _TRAIN_VAL_ARCHIVES = {
        "all": ("AID.tar", "8b69a4231c82e4605158268f01a0cd32a2eda8e99569b7b2ccb2a2b4dc50c68c"),
    }

    def decompress_integrity_check(self, decom_dir):
        data_dir = os.path.join(decom_dir, 'images')
        splits_dir = os.path.join(decom_dir, 'splits')

        if not os.path.exists(data_dir) or not os.path.exists(splits_dir):
            return False

        num_imgs = len(glob.glob(data_dir + '/*/*'))
        num_files = len(glob.glob(splits_dir + '/*'))

        return num_imgs == _NUM_IMAGES and num_files == _NUM_FILES

    def _decompress_dir(self):
        file_name, sha256 = self._TRAIN_VAL_ARCHIVES['all']
        if not self.decompress_integrity_check(self.decom_dir):
            print('Decompressing the tar file...')
            with tarfile.open(os.path.join(self.root, file_name), 'r') as tar:
                tar.extractall(self.decom_dir)
                tar.close()

    def _resources(self) -> List[OnlineResource]:
        file_name, sha256 = self._TRAIN_VAL_ARCHIVES['all']
        archive = HttpResource("https://syncandshare.lrz.de/dl/fiQzLLinL4N77kzcgJUhK4Cu/AID.tar", sha256=sha256)

        return [archive]

    def _prepare_sample(self, idx):
        img_path, cls_idx = self.data_item[idx]
        img_info = dict({'filename': img_path, 'cls_idx': cls_idx})

        return img_info

    def _prepare_list(self):
        self.data_item = []
        with open(os.path.join(self.decom_dir, 'splits', '{}_split.txt'.format(self._split))) as f:
            for line in f.readlines():
                img_name, cls_idx = line.strip().split(' ')
                self.data_item.append((img_name, int(cls_idx)))

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        self._decompress_dir()
        self._prepare_list()

        dp = SequenceWrapper(range(self.__len__()))
        # if not self.data_info:
        #     ndp = Mapper(dp, self._prepare_sample)
        #     ndp = hint_shuffling(ndp)
        #     ndp = hint_sharding(ndp)
        #     tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
        #                          transforms.RandomVerticalFlip(),
        #                          transforms.RandomResizedCrop((128, 128), scale=[0.5, 1]))
        #     ndp = ndp.map(tfs)
        # else:
        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:
        # return {
        #     'train': _TRAIN_LEN,
        #     'val': _VAL_LEN
        # }[self._split]
        return len(self.data_item)

if __name__ == '__main__':
    dp = Landslide4Sense('./')
