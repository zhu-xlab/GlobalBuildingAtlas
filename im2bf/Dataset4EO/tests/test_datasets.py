from Dataset4EO import datasets
import torch
from typing import Any, Dict, List, Tuple, Union, BinaryIO
from Dataset4EO.features import EncodedData
from Dataset4EO.utils._internal import fromfile, ReadOnlyTensorBuffer
import platform
import io
import mmap
import contextlib
import os
import numpy as np

import tempfile

tr = datasets.landcover_sen2.LandCoverSen2('./', split='train')
tr = tr.to_map_datapipe()

from torch.utils.data import DataLoader2
datas = DataLoader2(tr, batch_size=4, num_workers=4, shuffle=True)
import time

print('Loading Dataset...')

for eit in range(20):
    t = time.time()
    for dt in datas:
        pass
    t2 = time.time()
    print(t2 - t)

