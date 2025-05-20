import sys
import os
import pdb
from Dataset4EO.datasets import MunichBuilding
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2

datasets_dir = '../../Datasets/Dataset4EO/MunichBuilding'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = MunichBuilding(datasets_dir, split='train')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    for i, it in enumerate(dp):
        print(f'iter {i}: {it}')
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))
