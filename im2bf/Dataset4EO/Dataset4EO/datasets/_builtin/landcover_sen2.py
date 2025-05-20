import pathlib
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union, IO, Iterator
import enum
import functools
import torch
import io
import os
import mmap
import h5py
import numpy as np

from torchdata.datapipes.iter import IterDataPipe, Mapper, Demultiplexer, Filter, IterKeyZipper
from Dataset4EO.datasets.utils import Dataset, HttpResource, OnlineResource
from Dataset4EO.datasets.utils._internal import hint_sharding, hint_shuffling, INFINITE_BUFFER_SIZE, H5pyDataPipe

from Dataset4EO.features import BoundingBox, Label, EncodedImage
from .._api import register_dataset, register_info

NAME = "landcover"


@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(
        categories=(
            "Wheat", 
            "Barely",
            "Canola",
            "Lucerne/Medics",
            "Small grain grazing"
        )
    )



@register_dataset(NAME)
class LandCoverSen2(Dataset):
    """LandCover Sentinal Africa Dataset.
    homepage="https://www.esa.int",
    """

    def __init__(self, root: Union[str, pathlib.Path], *, split: str = "train", skip_integrity_check: bool = False) -> None:
        self._categories = _info()["categories"]
        self._split = split
        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:
        return [
            HttpResource(
                #"https://radiantearth.blob.core.windows.net/mlhub/archives/dlr_fusion_competition_germany_train_source_sentinel_2.tar.gz",
                #"https://radiantearth.blob.core.windows.net/mlhub/archives/ref_fusion_competition_south_africa_test_source_sentinel_2.tar.gz",
                "https://syncandshare.lrz.de/dl/fiLurHQ9Cy4NwvmPGYQe7RWM/landslide4sense.tar",
            )
        ]

    def _is_in_folder(self, data: Tuple[str, Any], *, name: str, depth: int = 1) -> bool:
        path = pathlib.Path(data[0])
        in_folder =  name in str(path.parent)
        return in_folder

    class _Demux(enum.IntEnum):
        TRAIN_IMG = 0
        TRAIN_MASK = 1
        VAL_IMG = 2

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        if self._is_in_folder(data, name="train/img", depth=2):
            return self._Demux.TRAIN_IMG
        if self._is_in_folder(data, name="train/mask", depth=2):
            return self._Demux.TRAIN_MASK
        elif self._is_in_folder(data, name="val/img", depth=2):
            return self._Demux.VAL_IMG
        else:
            return None    
    

    def _prepare_img(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        path, image = data
        return dict(
            path=path,
            img = image,
        )


    # collate_fn
    def _prepare_sample(self, data: Tuple[str, Any]) -> Dict[str, Any]:
        key, databuffers = data
        buffers, maskbuffers = databuffers
        pathmask, mask = maskbuffers
        pathimg, image = buffers
        image = h5py.File(pathimg.replace('.tar',''),'r')['img']
        mask = h5py.File(pathmask.replace('.tar',''),'r')['mask']
        return dict(
            imgpath=pathimg,
            maskpath=pathmask,
            #img = np.array(image),
            #mask = np.array(mask),
        )


    def _key_img(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        return path.stem[5:]

    def _key_mask(self, data: Tuple[str, Any]) -> str:
        path = pathlib.Path(data[0])
        return path.stem[4:]
        
    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:
        archive_dp = resource_dps[0]
        train_img_dp, train_mask_dp, val_dp = Demultiplexer(
            archive_dp,
            3,
            self._classify_archive,
            drop_none=True,
            buffer_size=INFINITE_BUFFER_SIZE,
        )

        if self._split == "train":
            dp = train_img_dp.zip_with_iter(train_mask_dp, key_fn=self._key_img, ref_key_fn=self._key_mask, buffer_size=INFINITE_BUFFER_SIZE, keep_key=True)
            fdp = Mapper(dp, self._prepare_sample)
        elif self._split == 'val':
            dp = H5pyDataPipe(val_dp, 'img')
            fdp = Mapper(dp, self._prepare_img)
        fdp = hint_shuffling(fdp)
        return fdp

    def __len__(self) -> int:
        if self._split=='train':
            return 3799
        else:
            return 245 

