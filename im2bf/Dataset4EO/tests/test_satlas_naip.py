import sys
import os
import pdb
from Dataset4EO.datasets import SatlasNAIP
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm

datasets_dir = '../Datasets/RS_datasets/NAIP'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = SatlasNAIP(datasets_dir, split='train_10k')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    img_sum = np.zeros(3)
    img_sum_squ = np.zeros(3)
    num_ite = 10000

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


