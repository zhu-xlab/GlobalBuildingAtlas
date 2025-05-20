import sys
import os
import pdb
from Dataset4EO.datasets import PlanetBuildingPairedV2
# from torch.utils.data import DataLoader2
#from torchdata.dataloader2 import DataLoader2 as DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2
import tifffile

num_bands = 4
datasets_dir = '../../Datasets/Dataset4EO/Planet'
ignore_list_path = None

"""
Dirs for the training split
"""
# window_dir = '../../Datasets/Dataset4EO/sr_training/sr_training/*/data'
# window_to_copy_dirs = [
#     ['../../Datasets/Dataset4EO/sr_training/sr_training/*/seg', {255:0, 0:1}],
#    '../../Datasets/Dataset4EO/Planet/pred_ndsm/sr_training/*/ndsm',
# ]
# split='train'
# crop_size=[256, 256]
# stride=[192, 192]
# collect_keys = ['img', 'ndsm']
# ignore_shp = True
# shape = (256, 256)
# use_ndsm=False
# use_ndsm_resource=False
# window_overide=False

"""
Dirs for the test split
"""
# window_dir = '../../Datasets/Dataset4EO/sr_training/sr_test/data'
# window_to_copy_dirs = [
#     '../../Datasets/Dataset4EO/sr_training/sr_test/seg',
#     '../../Datasets/Dataset4EO/Planet/pred_ndsm/sr_test/ndsm',
# ]
# split='test'
# crop_size=[256, 256]
# stride=[192, 192]
# collect_keys = ['img', 'seg']
# ignore_shp = True
# shape = (256, 256)
# use_ndsm=False
# use_ndsm_resource=False

"""
Test without window folder given
"""
# window_dir = None
# window_to_copy_dirs = []
# split='test'
# crop_size=[9600 * 2, 9600 * 2]
# stride=[9600 * 2, 9600 * 2]
# shape = (9600 * 2, 9600 * 2)
# ignore_shp=True
# collect_keys = ['img']
# use_ndsm_resource=False

"""
For the new test set
"""
raster_root = 'sr_4bands'
window_dir = [
    '../../Datasets/Dataset4EO/sr_training/sr_test/data',
    # '../../Datasets/Dataset4EO/SiningPlanet/bf_test_data/data',
    '../../Datasets/Dataset4EO/Planet/sr_test_gt_v2/*/*',
    # '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/bf_test_data_negative/*'
]
window_to_copy_dirs = [
    ['../../Datasets/Dataset4EO/sr_training/sr_test/seg', {255:0, 0:1}],
    # '../../Datasets/Dataset4EO/SiningPlanet/bf_test_data/seg',
    '../../Datasets/Dataset4EO/Planet/sr_test_gt_v2/*/seg',
    # '../../Datasets/Dataset4EO/SiningPlanet/ndsm',
    '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/sr_test/ndsm',
    '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/extra_test_filtered/ndsm',
    # '../../Datasets/Dataset4EO/Planet/pred_ndsm/sr_test/ndsm',
]
split='test'
crop_size=[256, 256]
stride=[192, 192]
collect_keys = [f'{num_bands}bands_img', 'seg', 'ndsm', 'majority_voting']
ignore_shp = True
shape = (256, 256)
use_ndsm=False
window_overide=False
additional_raster_resource_names = [('majority_voting', {255:1, 0:0})]
# additional_raster_resource_names = None

# new train
"""
window_dir = [
    '../../Datasets/Dataset4EO/sr_training/sr_training/*/data',
    '../../Datasets/Dataset4EO/Planet/sr_train_gt_sining_3-cities/seg',
]
window_to_copy_dirs = [
    ['../../Datasets/Dataset4EO/sr_training/sr_training/*/seg', {255:0, 0:1}],
    '../../Datasets/Dataset4EO/Planet/sr_train_gt_sining_3-cities/seg',
   '../../Datasets/Dataset4EO/Planet/pred_ndsm_v2/sr_training/*/ndsm',
]
raster_dir_name='sr_4bands'
split='train'
crop_size=[256, 256]
stride=[192, 192]
collect_keys = ['4bands_img', 'seg', 'ndsm']
ignore_shp = True
shape = (256, 256)
use_ndsm=False
window_overide=False
additional_raster_resource_names = None
ignore_list_path = '../../Datasets/Dataset4EO/Planet/list/high_loss_list.txt'
# additional_raster_resource_names = [('majority_voting', {255:1, 0:0})]
"""

"""
# test full images
raster_dir_name = 'negative_selected'
window_dir = None
window_to_copy_dirs = None
window_overide=False
split='test'
crop_size=[9600 * 2, 9600 * 2]
stride=[9600 * 2, 9600 * 2]
collect_keys = ['4bands_img', 'majority_voting']
ignore_shp = True
shape = (9600 * 2, 9600 * 2)
use_ndsm=False
use_ndsm_resource=False
# additional_raster_resource_names = None
additional_raster_resource_names = [('majority_voting', {255:1, 0:0})]
"""



from torchdata.dataloader2 import MultiProcessingReadingService

if __name__ == '__main__':
    img_sum = np.zeros(num_bands)
    img_sum_squ = np.zeros(num_bands)

    ndsm_sum = np.zeros(1)
    ndsm_sum_squ = np.zeros(1)

    num_ite = 0
    num_failed = 0

    dp = PlanetBuildingPairedV2(
        datasets_dir,
        collect_keys=collect_keys,
        raster_dir_name=raster_dir_name,
        window_root=window_dir,
        window_to_copy_dirs=window_to_copy_dirs,
        window_overide=window_overide,
        num_bands=num_bands,
        split=split,
        crop_size=crop_size,
        stride=stride,
        ignore_shp=ignore_shp,
        additional_raster_resource_names=additional_raster_resource_names,
        ignore_list_path=ignore_list_path
    )
    # data_loader = DataLoader2(dp.shuffle(), batch_size=1, num_workers=1, shuffle=True,
    #                           drop_last=True)
    city_cnt = {}
    continent_cnt = {}
    t1 = time.time()
    for i, it in enumerate(tqdm(dp)):
        # print(it)
        if not it['city_name'] in city_cnt:
            city_cnt[it['city_name']] = 0
        city_cnt[it['city_name']] += 1

        if not it['continent_name'] in continent_cnt:
            continent_cnt[it['continent_name']] = 0
        continent_cnt[it['continent_name']] += 1

        img_path = it['img_path']
        img = tifffile.imread(img_path).astype(np.float32)
        """
        if img.shape[:2] != shape:
            pdb.set_trace()
            print(img_path)

        """
        # if img.shape[-1] != num_bands:
        #     pdb.set_trace()
        #     num_failed += 1
        #     continue

        img_sum += img.sum(axis=0).sum(axis=0)
        img_sum_squ += (img ** 2).sum(axis=0).sum(axis=0)
        num_ite += 1

        if 'ndsm_path' in it and it['ndsm_path']:
            ndsm = tifffile.imread(it['ndsm_path'])
            ndsm_sum += ndsm.sum(axis=0).sum(axis=0)
            ndsm_sum_squ += (ndsm ** 2).sum(axis=0).sum(axis=0)

        # if num_ite % 1000 == 0:
        #     print(it)

        # print(f'iter {i}: {it}')

    t2 = time.time()
    print('loading time: {}'.format(t2-t1))
    print(f'number of iterations: {num_ite + num_failed}')
    print(f'success rate: {num_ite / (num_ite + num_failed)}')

    num_pixel = shape[0] * shape[1] * num_ite
    mean = img_sum / num_pixel
    std = ((img_sum_squ - 2 * img_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5
    print(f'img mean: {mean}')
    print(f'img stdm: {std}')

    mean = ndsm_sum / num_pixel
    std = ((ndsm_sum_squ - 2 * ndsm_sum * mean + mean ** 2 * num_pixel) / num_pixel) ** 0.5

    print(f'ndsm mean: {mean}')
    print(f'ndsm stdm: {std}')


    for key, value in city_cnt.items():
        print(f'{key} city has {value} patches')

    print('\n')

    for key, value in continent_cnt.items():
        print(f'{key} continent has {value} patches')
