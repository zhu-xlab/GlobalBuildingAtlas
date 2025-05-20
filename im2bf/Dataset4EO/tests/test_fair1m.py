import sys
import os
import pdb
from Dataset4EO.datasets import fair1m
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm

datasets_dir = '../Datasets/Bench_datasets/Fair1m'
from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    dp = fair1m.Fair1M(datasets_dir, split='test')
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    t1 = time.time()
    for i, it in enumerate(dp):
        print(f'iter {i}: {it}')
    t2 = time.time()
    print('loading time: {}'.format(t2-t1))


