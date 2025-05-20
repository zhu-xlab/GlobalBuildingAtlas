import sys
import os
import pdb
from Dataset4EO.datasets import SeasonNet
from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import tifffile
import numpy as np

datasets_dir = '../../Datasets/Dataset4EO/SeasonNet'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = SeasonNet(datasets_dir, split='test_1k', season='fall')
    data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
                              drop_last=False)

    img_sum = np.zeros(3)
    img_sum_squ = np.zeros(3)
    num_ite = 10000
    for epoch in range(1):
        t1 = time.time()
        for i, item in enumerate(tqdm(data_loader)):
            img = tifffile.imread(item['filename']).astype(np.float32)
            # img_sum += img.sum(axis=0).sum(axis=0)
            # img_sum_squ += (img ** 2).sum(axis=0).sum(axis=0)

        t2 = time.time()
        print('loading time: {}'.format(t2-t1))

    # num_pixel = len(data_loader) * (128 * 128)
    num_pixel = 120 * 120 * num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    print(mean)
    print(std)


