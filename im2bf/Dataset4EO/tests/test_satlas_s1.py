import sys
import os
import pdb
from Dataset4EO.datasets import SatlasS1
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2

datasets_dir = '../Datasets/RS_datasets/Sentinel'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = SatlasS1(datasets_dir, split='train')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    img_sum = np.zeros(2)
    img_sum_squ = np.zeros(2)
    num_ite = 10000

    t1 = time.time()
    for i, it in enumerate(dp):
        if i == num_ite:
            break

        vh_path, vv_path = it['filename']
        vh = cv2.imread(vh_path) * 1.0
        vv = cv2.imread(vv_path) * 1.0

        img_sum[0] += vh[:, :, 0].sum()
        img_sum_squ[0] += (vh[:, :, 0] ** 2).sum()

        img_sum[1] += vv[:, :, 0].sum()
        img_sum_squ[1] += (vv[:, :, 0] ** 2).sum()

        print(f'iter {i}: {it}')
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))

    num_pixel = 512 * 512 * num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    path = 'statistics/satlas_s1.npz'
    np.savez(path, mean=mean, std=std)
    print(f'mean and std values saved to {path}.')
