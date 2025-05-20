import os
import pdb
import numpy as np
import shutil
import zipfile
import rasterio

def copy_and_zip_files(file_paths, seg_paths, dest_folder, output_zip):

    data_folder = os.path.join(dest_folder, 'data')
    seg_folder = os.path.join(dest_folder, 'seg')

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        os.makedirs(seg_folder)

    for data_path, seg_path in zip(file_paths, seg_paths):
        seg_name = seg_path.split('/')[-1]
        shutil.copy(data_path, data_folder)

        raster = rasterio.open(data_path)
        out_seg_path = os.path.join(seg_folder, seg_name)
        seg = rasterio.open(seg_path).read()[0]

        with rasterio.open(
            out_seg_path, 'w',
            **raster.meta
        ) as dst:
            dst.write(seg, 1)  # Write array to the first band

    # Create a zip file and add all files from the destination folder
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(dest_folder):
            for filename in filenames:
                # Create the path of file to add to the zip
                path_to_file = os.path.join(foldername, filename)
                # Add file to the zip file with its path relative to the destination folder
                zipf.write(path_to_file, os.path.relpath(path_to_file, dest_folder))

    print(f"Created zip file {output_zip}")

file_paths, seg_paths, losses = [], [], []
list_path = 'outputs/temp/loss.txt'
dest_folder = 'outputs/temp2'
output_data_zip = 'outputs/temp2/filtered_data.zip'
output_seg_zip = 'outputs/temp2/filtered_seg.zip'
out_list = 'outputs/temp2/list.txt'
num_export = 10000

with open(list_path) as f:
    for line in f.readlines():
        img_path, seg_path, loss = line.split(' ')
        file_paths.append(img_path)
        seg_paths.append(seg_path)
        losses.append(float(loss))

idxes = np.argsort(losses)[::-1]
sorted_data_paths = [file_paths[x] for x in idxes]
sorted_seg_paths = [seg_paths[x] for x in idxes]
sorted_losses = [losses[x] for x in idxes]

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

copy_and_zip_files(
    sorted_data_paths[:num_export],
    sorted_seg_paths[:num_export],
    dest_folder, output_seg_zip
)
with open(out_list, 'w') as f:
    for data_path, seg_path, loss in zip(sorted_data_paths[:num_export],
                                         sorted_seg_paths[:num_export], sorted_losses[:num_export]):
        f.write(f'{data_path} {seg_path}, {loss}\n')
