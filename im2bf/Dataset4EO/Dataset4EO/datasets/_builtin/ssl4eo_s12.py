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
    Concater
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

NAME = "ssl4eo_s12"
_TRAIN_LEN = 100 # an example subset with 100 geo patches

'''
### no labels
@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))
'''

class SSL4EOResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        """
        # Download example patches
        wget https://drive.google.com/file/d/1HrUk-qxD9D8rqIhvvPFAisH7kBkrmcmA/view?usp=sharing
        """
        super().__init__('Download the data into the root directory',**kwargs)


@register_dataset(NAME)
class SSL4EO_S12(Dataset):
    """
    - **github link**: https://github.com/zhu-xlab/SSL4EO-S12
    - read from h5 file (each geo patch is one file), to be developed
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        data_info: bool = True,
        skip_integrity_check: bool = False,
    ) -> None:

        self.root = root
        self.decom_dir = os.path.join(self.root, 'ssl4eo_s12_h5_100')
        self.data_info = data_info

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _CHECKSUMS = {
        'ssl4eo_s12_h5_100.zip': 'B10B3407CE04393051A8886952621F145469D115D09D69B9A256D14B403B9C6F', # sha256
    }

    def get_classes(self):
        return self._categories

    def _resources(self) -> List[OnlineResource]:
        train_resource = SSL4EOResource(
            file_name = 'ssl4eo_s12_h5_100.zip',
            preprocess = 'extract',
            sha256 = self._CHECKSUMS['ssl4eo_s12_h5_100.zip']
        )
        return [train_resource]

    def _prepare_sample_dp(self, idx):
        iname_s1 = "{}/s1/{:07}.h5".format(self.decom_dir, idx)
        iname_s2a = "{}/s2a/{:07}.h5".format(self.decom_dir, idx)
        iname_s2c = "{}/s2c/{:07}.h5".format(self.decom_dir, idx)

        img_info = dict({'s1':iname_s1, 's2a':iname_s2a, 's2c':iname_s2c})
        return img_info

    def _prepare_sample(self, idx):

        iname_s1 = "{}/s1/{:07}.h5".format(self.decom_dir, idx)
        iname_s2a = "{}/s2a/{:07}.h5".format(self.decom_dir, idx)
        iname_s2c = "{}/s2c/{:07}.h5".format(self.decom_dir, idx)

        img_s1 = h5py.File(iname_s1, 'r')['array'][()] # [4, 2, 264, 264]
        img_s2a = h5py.File(iname_s2a, 'r')['array'][()] # [4, 12, 264, 264]
        img_s2c = h5py.File(iname_s2c, 'r')['array'][()] # [4, 13, 264, 264]

        ## use only the first season
        img_s1 = torch.tensor(img_s1[0])
        img_s2a = torch.tensor(img_s2a[0])
        img_s2c = torch.tensor(img_s2c[0])

        img_info = dict({'s1':iname_s1, 's2a':iname_s2a, 's2c':iname_s2c})

        return (img_info, img_s1, img_s2a, img_s2c)


    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        dp = SequenceWrapper(range(1, self.__len__()+1))
        if not self.data_info:
            ndp = Mapper(dp, self._prepare_sample)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)
            '''
            tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.RandomResizedCrop((256, 256), scale=[0.5, 1]))            
            ndp = ndp.map(tfs)
            '''
        else:
            ndp = Mapper(dp, self._prepare_sample_dp)
            ndp = hint_shuffling(ndp)
            ndp = hint_sharding(ndp)

        return ndp

    def __len__(self) -> int:

        return 100

if __name__ == '__main__':
    dp = SSL4EO_S12('/mnt/codes/datasets/ssl4eo-s12/ssl4eo_s12_100_patches/example_100_patches/ssl4eo-s12_h5')
