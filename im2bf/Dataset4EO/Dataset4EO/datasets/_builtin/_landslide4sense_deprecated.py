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
from torch.utils.data import DataLoader2
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

NAME = "landslide4sense"
_TRAIN_LEN = 3799
_VAL_LEN = 245


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

        # There is currently no test split available
        assert split != 'test'

        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))
        self.root = root
        self.decom_dir = os.path.join(self.root, 'landslide4sense')
        self._categories = _info()["categories"]
        self.CLASSES = ('background', 'landslide')
        self.PALETTE = [[128, 0, 0], [0, 128, 0]]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _TRAIN_VAL_ARCHIVES = {
        "trainval": ("landslide4sense.tar", "c7f6678d50c7003eba47b3cace8053c9bfa6b4692cd1630fe2d6b7bec11ccc77"),
    }

    def decompress_integrity_check(self, decom_dir):
        train_img_dir = os.path.join(decom_dir, 'train', 'img')
        train_mask_dir = os.path.join(decom_dir, 'train', 'mask')
        val_img_dir = os.path.join(decom_dir, 'val', 'img')

        if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir) or not os.path.exists(val_img_dir):
            return False

        num_train_img = len(os.listdir(train_img_dir))
        num_train_mask = len(os.listdir(train_mask_dir))
        num_val_img = len(os.listdir(val_img_dir))

        return (num_train_img == _TRAIN_LEN) and \
                (num_train_mask == _TRAIN_LEN) and \
                (num_val_img == _VAL_LEN)

    def _decompress_dir(self):
        file_name, sha256 = self._TRAIN_VAL_ARCHIVES['trainval']
        if not self.decompress_integrity_check(self.decom_dir):
            print('Decompressing the tar file...')
            with tarfile.open(os.path.join(self.root, file_name), 'r:gz') as tar:
                tar.extractall(self.decom_dir)
                tar.close()

    def _resources(self) -> List[OnlineResource]:
        file_name, sha256 = self._TRAIN_VAL_ARCHIVES['trainval']
        archive = HttpResource("https://syncandshare.lrz.de/getlink/fiLurHQ9Cy4NwvmPGYQe7RWM/{}".format(file_name), sha256=sha256)
        return [archive]

    def _prepare_sample_dp(self, idx):
        iname = "{}/img/image_{}.h5".format(self._split, idx)
        image_path = os.path.join(self.decom_dir, iname)
        label_path = None

        if self._split == 'train':
            mname = "{}/mask/mask_{}.h5".format(self._split, idx)
            label_path = os.path.join(self.decom_dir, mname)

        img_info = dict({'filename':image_path, 'ann':dict({'seg_map':label_path})})
        return img_info

    def _prepare_sample(self, idx):
        iname = "{}/img/image_{}.h5".format(self._split, idx)
        image_path = os.path.join(self.decom_dir, iname)
        img = h5py.File(os.path.join(self.decom_dir, iname), 'r')['img'][()]
        img = torch.tensor(img).permute(2, 0, 1)
        label_path = None
        img_info = dict({'filename':image_path, 'ann':dict({'seg_map':label_path})})

        if self._split == 'train':
            mname = "{}/mask/mask_{}.h5".format(self._split, idx)
            label_path = os.path.join(self.decom_dir, mname)
            mask = h5py.File(os.path.join(self.decom_dir, mname), 'r')['mask'][()]
            mask = torch.tensor(mask)
            img_info = dict({'filename':image_path, 'ann':dict({'seg_map':label_path})})
            return (img_info, img, mask)

        return (img_info, img)

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        self._decompress_dir()
        dp = SequenceWrapper(range(1, self.__len__()+1))
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
            'val': _VAL_LEN
        }[self._split]

if __name__ == '__main__':
    dp = Landslide4Sense('./')
