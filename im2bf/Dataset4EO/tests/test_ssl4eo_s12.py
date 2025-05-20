import sys
import os
import pdb
from Dataset4EO.datasets import SSL4EO_S12
from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm

datasets_dir = '/mnt/d/codes/datasets/ssl4eo-s12/ssl4eo_s12_100_patches/example_100_patches/ssl4eo-s12_h5'
from torchdata.dataloader2 import MultiProcessingReadingService


if __name__ == '__main__':
    dp = SSL4EO_S12(datasets_dir)
    data_loader = DataLoader2(dp, batch_size=4, num_workers=4, shuffle=True, drop_last=True)
    for epoch in range(5):
        t1 = time.time()
        for it in tqdm(data_loader):
            print(it)
            pass
        t2 = time.time()
        print(t2-t1)