import sys
import os
import pdb
from Dataset4EO.datasets import SatlasS2
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2

datasets_dir = '../Datasets/RS_datasets/Sentinel'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = SatlasS2(datasets_dir, split='train')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    img_sum = np.zeros(9)
    img_sum_squ = np.zeros(9)
    num_ite = 100000

    for i, it in enumerate(dp):
        if i == num_ite:
            break

        print(f'iter {i}: {it}')
        for i, path in enumerate(it['filename']):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 1.0

            if i == 0: # rgb image
                img_sum[:3] += img_rgb.sum(axis=0).sum(axis=0)
                img_sum_squ[:3] += (img_rgb ** 2).sum(axis=0).sum(axis=0)
            else:
                img_sum[i+2] += img_rgb[:, :, 0].sum()
                img_sum_squ[i+2] += (img_rgb[:, :, 0] ** 2).sum()



    t2 = time.time()
    print('loading time: {}'.format(t2-t1))

    num_pixel = 512 * 512 * num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    path = 'statistics/satlas_s2.npz'
    np.savez(path, mean=mean, std=std)
    print(f'mean and std values saved to {path}.')

