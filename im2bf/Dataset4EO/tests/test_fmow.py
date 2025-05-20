import sys
import os
import pdb
from Dataset4EO.datasets import FMoW
from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import tifffile
import numpy as np

datasets_dir = '../../Datasets/Dataset4EO/fMoW'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = FMoW(datasets_dir, split='val')
    data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=4, shuffle=True,
                              drop_last=False)

    for epoch in range(1):
        t1 = time.time()
        for i, item in enumerate(tqdm(data_loader)):
            print(item)

        t2 = time.time()
        print('loading time: {}'.format(t2-t1))


