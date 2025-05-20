import sys
import os
import pdb
from Dataset4EO.datasets import PlanetBuildingPaired
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2

datasets_dir = '../../Datasets/Dataset4EO/PlanetBuildingV2'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = PlanetBuildingPaired(
        datasets_dir, split='train', city_names=['berlin'],
        img_type = 'building_footprint',
        mask_type='osm_3m',
        gdal_poly_type='microsoft_polygon',
        gt_poly_type='osm_polygon',
        crop_size=[256, 256],
        stride=[192, 192],
        mask_scale=1,
        # crop_size=[-1,-1],
        # stride=[-1,-1]
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
