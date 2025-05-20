import os
#import tarfile
#import enum
#import functools
import pathlib
#from tqdm import tqdm
#import h5py
#import torch
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union
#from xml.etree import ElementTree
#from torch.utils.data import DataLoader2
from Dataset4EO import transforms
import pdb
#import numpy as np

from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    FileLister
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

NAME = "geonrw"
_TRAIN_LEN = 6942
_TEST_LEN = 463


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class GeoNRWResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Download data from https://ieee-dataport.org/open-access/geonrw.", **kwargs)


@register_dataset(NAME)
class GeoNRW(Dataset):
    """
    - **homepage**: https://ieee-dataport.org/open-access/geonrw
    - prepare dataset:
        - download and extract data from https://ieee-dataport.org/open-access/geonrw
        - run `tools/convert_datasets/geonrw.py` to reorganize the data w.r.t mmsegmentation
            - nrw_dataset_mmseg/
                - ann_dir
                    - train
                    - test
                - dem_dir
                    - train
                    - test
                - img_dir
                    - train
                    - test
    
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
        self.dir_name = 'nrw_dataset_mmseg'
        self.data_prefix = os.path.join(self.root,self.dir_name)
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [
        [44, 160, 44],
        [31, 119, 180],
        [140, 86, 75],
        [127, 127, 127],
        [188,189,34],
        [255,127,14],
        [148,103,189],
        [23,190,207],
        [214,39,40],
        [227, 119, 194]
        ]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)


    def _resources(self) -> List[OnlineResource]:
        resource = GeoNRWResource(
            file_name = self.dir_name,
            preprocess = None,
            sha256 = None
        )
        return [resource]    

    def _prepare_sample(self, data):

        image_path, label_path = data
        img_info = dict({'filename':image_path, 'ann':dict({'seg_map':label_path})})

        return img_info

    def _is_in_folder(self, data: Tuple[str, Any], *, name: str, depth: int = 1) -> bool:
        path = pathlib.Path(data)
        in_folder =  name in str(path.parent)
        return in_folder

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        if self._is_in_folder(data, name="train", depth=2):
            return 0
        if self._is_in_folder(data, name="test", depth=2):
            return 1

        else:
            return None

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        image_dp = FileLister(root=os.path.join(self.data_prefix, 'img_dir'), recursive=True)
        train_image_dp, test_image_dp = image_dp.demux(num_instances=2, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)
        label_dp = FileLister(root=os.path.join(self.data_prefix, 'ann_dir'), recursive=True)
        train_label_dp, test_label_dp = label_dp.demux(num_instances=2, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)        
        #dem_dp = FileLister(root=os.path.join(self.data_prefix, 'dem_dir'), recursive=True)
        #train_dem_dp, test_dem_dp = label_dp.demux(num_instances=2, classifier_fn=self._classify_archive,\
        #        drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)

        train_dp = train_image_dp.zip(train_label_dp)
        test_dp = test_image_dp.zip(test_label_dp)
        
        ndp = eval(self._split+'_dp')
        ndp = Mapper(ndp, self._prepare_sample)
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)
        

        return ndp

    def __len__(self) -> int:
        return {
            'train': _TRAIN_LEN,
            'val': _VAL_LEN
        }[self._split]

if __name__ == '__main__':
    dp = GeoNRW('./')
