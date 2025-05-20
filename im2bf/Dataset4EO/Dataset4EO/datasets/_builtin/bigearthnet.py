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

NAME = "bigearthnet"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


class BigEarthNetResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:

        super().__init__('For data download, please refer to https://bigearth.net/',
                         **kwargs)


@register_dataset(NAME)
class BigEarthNet(Dataset):
    """
    - **homepage**: https://bigearth.net/
    - preparation:
        - download and extract `BigEarthNet-S2-v1.0.tar.gz`
        - run `tools/convert_datasets/bigearthnet_convert_rgb.py` to prepare RGB images and gather corresponding labels in `19_labels.csv`
            - `BigEarthNet-v1.0-RGB/`
            - `19_labels.csv`
        - download train/val/test split file
            - https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/train.csv?inline=false
            - https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/val.csv?inline=false
            - https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/master/splits/test.csv?inline=false
        - split the `19_labels.csv` w.r.t the splits
            - `19_labels_train.csv`
            - `19_labels_val.csv`
            - `19_labels_test.csv`
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:


        self._split = self._verify_str_arg(split, "split", ("train", "val", "test"))
        self.root = root
        self.dir_name = 'BigEarthNet-v1.0-RGB'
        self._categories = _info()["categories"]
        self.data_info = data_info
        self.data_prefix = os.path.join(self.root,self.dir_name)

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        resource_rgb = BigEarthNetResource(
            file_name = self.dir_name,
            preprocess = None,
            sha256 = None
        )
        resource_train = BigEarthNetResource(
            file_name = '19_labels_train.csv',
            preprocess = None,
            sha256 = None
        )        
        resource_val = BigEarthNetResource(
            file_name = '19_labels_val.csv',
            preprocess = None,
            sha256 = None
        )        
        resource_test = BigEarthNetResource(
            file_name = '19_labels_test.csv',
            preprocess = None,
            sha256 = None
        )        
        return [resource_rgb, resource_train, resource_val, resource_test]


    def _prepare_sample(self, idx):
        data_info = self.data_item[idx]
        
        return data_info

    def _prepare_list(self):
        self.data_item = []
        with open(os.path.join(self.root, '19_labels_{}.csv'.format(self._split))) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
            
            for sample in samples:
                filename = sample[0] + '.png'
                gt_label = sample[1:]
                target = np.zeros(19, dtype=np.uint8)
                for id in gt_label:
                    target[int(id)] = 1
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(target, dtype=np.uint8) # binary
                self.data_item.append(info)
                
               

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        self._prepare_list()

        dp = SequenceWrapper(range(self.__len__()))

        ndp = Mapper(dp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:

        return len(self.data_item)

if __name__ == '__main__':
    dp = BigEarthNet('./')
