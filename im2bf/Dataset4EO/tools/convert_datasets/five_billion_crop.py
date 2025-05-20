import os
import cv2
import pdb
from tqdm import tqdm
import re
import tifffile
import rasterio
import shutil


img_dir = '../Datasets/RS_datasets/Gaofen/FiveBillion/Image__8bit_NirRGB/'
img_dir_16bit = '../Datasets/RS_datasets/Gaofen/FiveBillion/Image_16bit_NirRGB/'
meta_dir = '../Datasets/RS_datasets/Gaofen/FiveBillion/Coordinate_files'
w_size = 512
stride = 256

res_dir = '../Datasets/RS_datasets/Gaofen/FiveBillion/images_splited_{}_{}'.format(w_size, stride)
res_dir_meta = '../Datasets/RS_datasets/Gaofen/FiveBillion/meta_data_splited_{}_{}'.format(w_size, stride)
splited_images_dir = os.path.join(res_dir, 'images')
splited_meta_dir = res_dir_meta

r = re.compile('[a-zA-Z\-]+')

if not os.path.exists(splited_images_dir):
    os.makedirs(splited_images_dir)

if not os.path.exists(splited_meta_dir):
    os.makedirs(splited_meta_dir)

for img_name in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    img_path_16bit = os.path.join(img_dir_16bit, img_name + 'f') # .tif -> .tiff

    splited_img_path_check = os.path.join(splited_images_dir, img_name.split('.')[0] + '_patch_00x00.tif')

    if os.path.exists(splited_img_path_check):
        continue
    # coord_path = os.path.join(coordinate_img_dir, img_name.split('.')[0] + '.rpb')
    # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    try:
        img = tifffile.imread(img_path)
        tif_16bit = tifffile.TiffFile(img_path_16bit)
    except Exception:
        print(f'corrupted file: {img_name}, skip it')
        continue

    profile = {}
    for tag in tif_16bit.pages[0].tags.values():
        profile[tag.name] = tag.value

    H, W, C = img.shape

    cur_x, cur_y = 0, 0
    cnt = 0
    for i in range((H-1) // stride + 1):
        cur_x = 0
        for j in range((W-1) // stride + 1):
            # pdb.set_trace()
            crop_img = img[cur_y:cur_y+w_size, cur_x:cur_x+w_size, :]
            splited_img_path = os.path.join(splited_images_dir, img_name.split('.')[0] +
                                            '_patch_{:0>2d}x{:0>2d}.tif'.format(i, j))

            tifffile.imsave(splited_img_path, crop_img, metadata=profile)
            cnt += 1
            if cur_x == W-w_size+1:
                break
            cur_x = min(cur_x + stride, W-w_size)

        if cur_y == H-w_size+1:
            break
        cur_y = min(cur_y + stride, H-w_size)

for img_name in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    splited_meta_path_check = os.path.join(splited_meta_dir, img_name.split('.')[0] + '_patch_00x00.tif')
    src_meta_path = os.path.join(meta_dir, img_name.replace('.tif', '.rpb'))

    if os.path.exists(splited_meta_path_check):
        continue
    try:
        img = tifffile.imread(img_path)
    except Exception:
        print(f'corrupted file: {img_name}, skip it')
        continue

    H, W, C = img.shape

    cur_x, cur_y = 0, 0
    cnt = 0
    for i in range((H-1) // stride + 1):
        cur_x = 0
        for j in range((W-1) // stride + 1):
            # pdb.set_trace()
            crop_img = img[cur_y:cur_y+w_size, cur_x:cur_x+w_size, :]
            splited_meta_path = os.path.join(splited_meta_dir, img_name.split('.')[0] \
                                             + '_patch_{:0>2d}x{:0>2d}.rbp'.format(i, j))
            shutil.copy(src_meta_path, splited_meta_path)

            cnt += 1
            if cur_x == W-w_size+1:
                break
            cur_x = min(cur_x + stride, W-w_size)

        if cur_y == H-w_size+1:
            break
        cur_y = min(cur_y + stride, H-w_size)
