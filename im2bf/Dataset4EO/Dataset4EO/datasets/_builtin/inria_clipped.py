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
import math
from ..utils import clip_big_image


from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    Concater,
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

NAME = "inria_clipped"
_TRAIN_LEN = 31 * 5 * 25
_VAL_LEN = 5 * 5 * 25
_TEST_LEN = 36 * 5 * 25

_TRAIN_CITY_NAMES = ['austin{}.tif', 'chicago{}.tif', 'kitsap{}.tif', 'vienna{}.tif', 'tyrol-w{}.tif']

_VAL_IMG_NAMES = []

for idx in range(1, 6, 1):
    for name in _TRAIN_CITY_NAMES:
        _VAL_IMG_NAMES.append(name.format(idx))


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class InriaResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://project.inria.fr/aerialimagelabeling/ and follow the instructions there.", **kwargs)


@register_dataset(NAME)
class InriaClipped(Dataset):
    """
    - **homepage**: https://project.inria.fr/aerialimagelabeling/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        city_names=None, # None -> use all cities
        data_info: bool = True,
        skip_integrity_check: bool = False,
        crop_size: int = 1024,
        stride: int = 512
    ) -> None:

        assert split in ('train', 'val', 'train_val', 'test')

        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[0,0,0], [255,255,255]]
        self.data_info = data_info
        self.crop_size = crop_size
        self.stride=stride
        self.city_names = city_names

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        "all": ("NEW2-AerialImageDataset.zip", "2e0ee5051b794e5cedc2c915bd839a29d8994fcca6d50727b031ab6bfbf4ef88"),
    }

    def _resources(self) -> List[OnlineResource]:
        resource = InriaResource(
            file_name = self._CHECKSUMS['all'][0],
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['all'][1]
        )

        return [resource]

    def _prepare_sample_dp(self, data):

        if self._split == 'test':
            image_path = data
            img_info = dict({'filename':image_path})

        else:
            image_path, ann_path = data
            img_info = dict({'filename':image_path, 'ann':dict({'seg_map':ann_path})})

        return img_info

    def _prepare_sample(self, data):

        if self._split == 'test':
            image_path = data
            img = EncodedImage.from_path(image_path).decode()
            return (key, img)

        else:
            image_path, ann_path = data
            img = EncodedImage.from_path(image_path).decode()
            ann = EncodedImage.from_path(ann_path).decode()

            return (key, img, ann)

    def _classify_split(self, data):
        path = pathlib.Path(data[0])
        if path.parents[1].name == 'train':
            for img_name in _VAL_IMG_NAMES:
                if img_name == path.name:
                    return 1

            return 0

        else:
            return 2

    def _filter_cities(self, data):
        import re

        path = pathlib.Path(data)
        name = re.findall('[a-z\-]+', path.name)[0]
        if self.city_names is None or name in self.city_names:
            return True
        return False


    def _classify_split_clip(self, data):
        path = pathlib.Path(data)
        if path.parents[0].name == 'train':
            return 0
        elif path.parents[0].name == 'val':
            return 1
        elif path.parents[0].name == 'test':
            return 2

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.parents[0].name == 'images':
            return 0
        elif path.parents[0].name == 'gt':
            return 1
        else:
            return None

    def _classify_archive_clip(self, data):
        path = pathlib.Path(data)
        if path.parents[1].name == 'img_dir':
            return 0
        elif path.parents[1].name == 'ann_dir':
            return 1
        else:
            raise ValueError()

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        clip_dir = self._clip_images(resource_dps)
        dp = FileLister(clip_dir, recursive=True)

        train_dp, val_dp, test_dp = Demultiplexer(
            dp, 3, self._classify_split_clip, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )
        dp = eval(f'{self._split}_dp') if self._split != 'train_val' else Concater(train_dp, val_dp)

        dp = Filter(dp, self._filter_cities)

        img_dp, ann_dp = Demultiplexer(
            dp, 2, self._classify_archive_clip, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )


        if self._split != 'test':
            dp = Zipper(img_dp, ann_dp)
        else:
            dp = img_dp

        if not self.data_info:
            dp = Mapper(dp, self._prepare_sample)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)
            tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomCrop((512, 512), scale=[0.5, 1]))
            ndp = ndp.map(tfs)
        else:
            ndp = Mapper(dp, self._prepare_sample_dp)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)

        return ndp


    def _clip_images(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        # dp = Filter(resource_dps[0], self._select_split)
        train_dp, val_dp, test_dp = Demultiplexer(
            resource_dps[0], 3, self._classify_split, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        clip_size=self.crop_size
        stride_size=self.stride
        img_W = 5000
        assert clip_size < img_W and clip_size > 0
        assert img_W > stride_size and stride_size > 0
        num_imgs = math.ceil((img_W - clip_size) / stride_size) + 1

        for split in ['train', 'val', 'test']:
            dp = eval(f'{split}_dp')

            img_dp, gt_dp = Demultiplexer(
                dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )
            clip_dir_img = os.path.join(self.root,
                                        'NEW2-AerialImageDataset',
                                        'clip_c{}_s{}'.format(clip_size, stride_size),
                                        'img_dir',
                                        split)

            clip_dir_ann = os.path.join(self.root,
                                        'NEW2-AerialImageDataset',
                                        'clip_c{}_s{}'.format(clip_size, stride_size),
                                        'ann_dir',
                                        split)

            if os.path.exists(clip_dir_img): # images already generated
                continue
            else:
                print(f'Clipping the Inria dataset to patches of sizes {clip_size} x {clip_size}. May take a while...')

            if split == 'train' or split == 'val':
                for img_path, gt_path in tqdm(zip(img_dp, gt_dp)):
                    img_path = img_path[0]
                    gt_path = gt_path[0]
                    clip_big_image(img_path, clip_dir_img, clip_size=clip_size,
                                   stride_size=stride_size)
                    clip_big_image(gt_path, clip_dir_ann, clip_size=clip_size,
                                   stride_size=stride_size, is_label=True)
            else:
                for img_path in tqdm(img_dp):
                    img_path = img_path[0]
                    clip_big_image(img_path, clip_dir_img, clip_size=clip_size,
                                   stride_size=stride_size)


        clip_dir = os.path.join(self.root,
                                'NEW2-AerialImageDataset',
                                'clip_c{}_s{}'.format(clip_size, stride_size))
        return clip_dir


    def __len__(self) -> int:
        base_len = {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN,
            'train_val': _TRAIN_LEN + _VAL_LEN,
            'test': _TEST_LEN

        }[self._split]
        if self.city_names is not None:
            base_len = int(base_len / 5 * len(self.city_names))

        return base_len

if __name__ == '__main__':
    dp = Landslide4Sense('./')
