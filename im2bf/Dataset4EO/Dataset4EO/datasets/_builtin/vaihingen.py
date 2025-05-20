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

NAME = "vaihingen"
_TRAIN_LEN = 8
_VAL_LEN = 8


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class VaihingenResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Download data from https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx.", **kwargs)


@register_dataset(NAME)
class Vaihingen(Dataset):
    """
    - **homepage**: https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        self._split = self._verify_str_arg(split, "split", ("train", "val"))
        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        "all": ("Vaihingen.zip", "ddde00345439f0a778a8fb4cf325c7791d9c66da4378fbcea44ca3ff7678fa0f"),
    }

    def _resources(self) -> List[OnlineResource]:
        resource = VaihingenResource(
            file_name = self._CHECKSUMS['all'][0],
            #preprocess = 'extract',
            sha256 = self._CHECKSUMS['all'][1]
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
        if self._is_in_folder(data, name="val", depth=2):
            return 1

        else:
            return None

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        image_dp = FileLister(root=os.path.join(self.root, 'img_dir'), recursive=True)
        train_image_dp, val_image_dp = image_dp.demux(num_instances=2, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)
        label_dp = FileLister(root=os.path.join(self.root, 'ann_dir'), recursive=True)
        train_label_dp, val_label_dp = label_dp.demux(num_instances=2, classifier_fn=self._classify_archive,\
                drop_none=True, buffer_size=INFINITE_BUFFER_SIZE)        


        train_dp = train_image_dp.zip(train_label_dp)
        val_dp = val_image_dp.zip(val_label_dp)

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
    dp = Vaihingen('./')
