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
    Zipper,
    IterKeyZipper,
    LineReader,
    JsonParser,
    IterableWrapper
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

NAME = "fmow"
_TRAIN_LEN = 1
_VAL_LEN = 2
_TEST_LEN = 2


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class FMoWResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download FMoW data manually at https://github.com/fMoW/dataset:
        """
        super().__init__('For data download, please refer to https://github.com/fMoW/dataset',
                         **kwargs)


@register_dataset(NAME)
class FMoW(Dataset):
    """
    - **homepage**: https://github.com/fMoW/dataset
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        use_ms = True,
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        assert split in ['train', 'val', 'test']
        self._split = split

        self.root = root
        self.use_ms = use_ms
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128]]
        self.data_info = data_info
        self.cat2idx = {name: idx for idx, name in enumerate(self.CLASSES)}

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        train_resource = FMoWResource(
            file_name='train',
            preprocess=None,
            sha256=None
        )
        val_resource = FMoWResource(
            file_name='val',
            sha256=None
        )
        test_resource = FMoWResource(
            file_name='test',
            preprocess=None,
            sha256=None
        )

        test_mapping_resource = FMoWResource(
            file_name='test_gt_mapping.json',
            preprocess=None,
            sha256=None
        )

        test_json_resource = FMoWResource(
            file_name='test_gt',
            preprocess=None,
            sha256=None
        )

        return [train_resource, val_resource, test_resource,
                test_mapping_resource, test_json_resource]

    def _prepare_sample(self, data):

        if self._split != 'test':
            img_data, ann_data = data
            img_path, img_buffer = img_data
            ann_path, ann_buffer = ann_data

            class_name = pathlib.Path(img_path).parents[1].name
            cls_idx = self.cat2idx[class_name]

            img_info = dict(filename=img_path,
                            cls_idx=cls_idx,
                            ann=dict(ann_path=ann_path))

        else:
            (img_path, img_buffer), (mapping) = data
            class_name = pathlib.Path(mapping['input']).parent.name
            cls_idx = self.cat2idx[class_name]
            img_info = dict(filename=img_path,
                            cls_idx=cls_idx,
                            ann=None)

        return img_info

    def _filter_img_path(self, data):
        img_path, img_buffer = data
        postfix = '_msrgb.jpg' if self.use_ms else '_rgb.jpg'

        return pathlib.Path(img_path).name.endswith(postfix)


    def _classify_archive(self, data):

        img_path, img_buffer = data
        postfix_rgb = '_rgb.jpg'
        postfix_msrgb = '_msrgb.jpg'
        postfix_json = '_rgb.json'
        postfix_msjson = '_msrgb.json'

        path_name = pathlib.Path(img_path).name
        if path_name.endswith(postfix_rgb):
            return 0
        elif path_name.endswith(postfix_msrgb):
            return 1
        elif path_name.endswith(postfix_json):
            return 2
        elif path_name.endswith(postfix_msjson):
            return 3

    def _test_key_fn(self, data):
        path, buffer = data
        path = pathlib.Path(path)

        return path.parent.name

    def _mapping_key_fn(self, mapping):
        path = pathlib.Path(mapping['output']).name

        return path


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        train_dp, val_dp, test_dp, test_mapping_dp, test_gt_dp = resource_dps

        if self._split != 'test':
            dp = eval(f'{self._split}_dp')
            rgb_dp, msrgb_dp, json_dp, msjson_dp = Demultiplexer(
                dp, 4, self._classify_archive, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
            )
            img_dp = rgb_dp if not self.use_ms else msrgb_dp
            ann_dp = json_dp if not self.use_ms else msjson_dp
            dp = Zipper(img_dp, ann_dp) if not self._split == 'test' else img_dp
        else:
            test_mapping_dp = JsonParser(test_mapping_dp)
            mapping = iter(test_mapping_dp).__next__()[1]
            mapping_dp = IterableWrapper(mapping)
            dp = IterKeyZipper(test_dp, mapping_dp, self._test_key_fn, self._mapping_key_fn,
                                    buffer_size=INFINITE_BUFFER_SIZE)

        ndp = Mapper(dp, self._prepare_sample)
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
