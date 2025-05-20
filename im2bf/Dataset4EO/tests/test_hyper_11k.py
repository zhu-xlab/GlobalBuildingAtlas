import sys
import os
import pdb
from Dataset4EO.datasets import Hyper11k
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import tifffile
import numpy as np

#datasets_dir = '../Datasets/Hyperspectral/'
datasets_dir = '../Datasets/RS_datasets/Hyperspectral/Hyper-11k/'
from torchdata.dataloader2 import MultiProcessingReadingService

invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160,
                    161, 162, 163, 164, 165, 166]
valid_channels_ids = [c for c in range(224) if c not in invalid_channels]
num_valid_channels = len(valid_channels_ids)
min_val = 0
max_val = 10000

if __name__ == '__main__':
    dp = Hyper11k(datasets_dir, split='train')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    img_sum = np.zeros(num_valid_channels)
    img_sum_squ = np.zeros(num_valid_channels)
    num_ite = 11483

    for i, it in tqdm(enumerate(dp)):
        if i == num_ite:
            break

        img = tifffile.imread(it['filename'])
        img = img[valid_channels_ids]

        clipped = np.clip(img, a_min=min_val, a_max=max_val)
        out_data = (clipped - min_val) / (max_val - min_val)
        out_data = out_data.astype(np.float32)

        img_sum += out_data.sum(axis=1).sum(axis=1)
        img_sum_squ += (out_data ** 2).sum(axis=1).sum(axis=1)


    num_pixel = 128 * 128 * num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    path = 'statistics/hyper_11k.npz'
    np.savez(path, mean=mean, std=std)
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))


