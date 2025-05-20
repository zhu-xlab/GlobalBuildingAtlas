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

NAME = "nwpu_vhr10"
_TRAIN_LEN = 3799
_VAL_LEN = 245


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class NWPU_VHR10Resource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download DIOR data manually:
        """
        super().__init__('For data download, please refer to https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC',
                         **kwargs)

@register_dataset(NAME)
class NWPU_VHR10(Dataset):
    """
    - **homepage**: https://github.com/chaozhong2010/VHR-10_dataset_coco
      Not finished.
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        self._split = self._verify_str_arg(split, "split", ("train", "test"))
        self.root = root
        self._categories = _info()["categories"]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        "NWPU VHR-10 dataset": 'a417f01609ff7cf723751e17c95075afb7debcb1d3b4ed8e06a2d99f6c8c4fb6',
    }

    def _resources(self) -> List[OnlineResource]:
        resource = HttpResource(
            file_name = 'NWPU VHR-10 dataset.rar',
            url = self._CHECKSUMS['NWPU VHR-10 dataset'][0],
            # preprocess = 'extract',
            sha256 = self._CHECKSUMS['all'][1]
        )

        resource = NWPU_VHR10Resource(
            file_name = 'ImageSets.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['ImageSets.zip']
        )

        return [resource]

    def _prepare_sample_dp(self, data):

        key, (image_data, ann_data) = data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img_info = dict({'filename':image_path, 'ann':dict({'seg_map':ann_path})})

        return img_info

    def _prepare_sample(self, data):

        key, (image_data, ann_data) = data
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        img = EncodedImage.from_path(image_path).decode()
        ann = EncodedImage.from_path(ann_path).decode()

        return (key, img, ann)

    def _select_split(self, data):
        path = pathlib.Path(data[0])
        return path.parents[1].name == self._split

    def _classify_archive(self, data):
        path = pathlib.Path(data[0])
        if path.parents[0].name == 'images':
            return 0
        elif path.parents[0].name == 'gt':
            return 1
        else:
            return None

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])
        return path.name


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        dp = Filter(resource_dps[0], self._select_split)
        img_dp, gt_dp = Demultiplexer(
            dp, 2, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )

        dp = IterKeyZipper(
            img_dp, gt_dp,
            key_fn=self._images_key_fn,
            ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True
        )

        if not self.data_info:
            dp = Mapper(dp, self._prepare_sample)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)
            tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomCrop((256, 256), scale=[0.5, 1]))
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
