import sys
import os
import pdb
from Dataset4EO.datasets import five_billion
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import tifffile

datasets_dir = '../Datasets/RS_datasets/Gaofen/FiveBillion'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = five_billion.FiveBillion(datasets_dir, split='train')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    img_sum = np.zeros(4)
    img_sum_squ = np.zeros(4)
    num_ite = 117450

    t1 = time.time()
    for i, it in enumerate(dp):
        if i == num_ite:
            break

        print(f'iter {i}: {it}')
        img = tifffile.imread(it['filename']) * 1.0
        img_sum += img.sum(axis=0).sum(axis=0)
        img_sum_squ += (img ** 2).sum(axis=0).sum(axis=0)

    t2 = time.time()
    print('loading time: {}'.format(t2-t1))

    num_pixel = 512 * 512 * num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    path = 'statistics/five_billion.npz'
    np.savez(path, mean=mean, std=std)
    print(f'mean and std values saved to {path}.')
