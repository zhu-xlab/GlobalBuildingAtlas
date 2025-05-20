import sys
import os
import pdb
from Dataset4EO.datasets import PlanetBuildingShape
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2

datasets_dir = '../../Datasets/Dataset4EO/PlanetBuildingV3/shapes'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = PlanetBuildingShape(
        datasets_dir
    )
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    for i, it in enumerate(dp):
        print(f'iter {i}: {it}')
        if i == 100:
            break
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))
