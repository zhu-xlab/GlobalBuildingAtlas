import sys
import os
import pdb
import tqdm
from Dataset4EO.datasets import SRTrainingDataset
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2
import tifffile

datasets_dir = '../../Datasets/Dataset4EO/sr_training'

if __name__ == '__main__':
    dp = SRTrainingDataset(
        datasets_dir, split='train',
    )

    img_sum = np.zeros(4)
    img_sum_squ = np.zeros(4)
    num_ite = 10000

    t1 = time.time()
    for i, item in enumerate(tqdm(dp)):
        img = tifffile.imread(item['img_info']['filename']).astype(np.float32)
        img_sum += img.sum(axis=0).sum(axis=0)
        img_sum_squ += (img ** 2).sum(axis=0).sum(axis=0)
        if i == num_ite:
            break

    t2 = time.time()
    print('loading time: {}'.format(t2-t1))

    # num_pixel = len(data_loader) * (128 * 128)
    num_pixel = 256 * 256* num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    print(mean)
    print(std)

