# import mmcv
import os.path as osp
import numpy as np
import math
import pdb

def clip_big_image(image_path, clip_save_dir,
                   clip_size=1024, stride_size=512,
                   is_label=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path)

    h, w, c = image.shape

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    # xmin = x * clip_size
    # ymin = y * clip_size
    xmin = x * stride_size
    ymin = y * stride_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ], axis=1)


    if is_label:
        image = image[:, :, 0]
        image[image==255] = 1

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        # clipped_image = image[start_y:end_y,
        #                       start_x:end_x] if to_label else image[
        #                           start_y:end_y, start_x:end_x, :]
        clipped_image = image[start_y:end_y, start_x:end_x, ...]
        # idx_i, idx_j = osp.basename(image_path).split('_')[2:4]
        img_name = osp.basename(image_path).split('.')[0]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{img_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))



