import sys
import os
import pdb
from Dataset4EO.datasets import dior
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm

datasets_dir = '../../Datasets/Dataset4EO/DIOR'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = dior.DIOR(datasets_dir, split='train')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    for it in dp:
        pass
        print(it)
        break
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))


