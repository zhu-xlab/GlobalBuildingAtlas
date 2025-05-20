import os
import geopandas as gpd
import pdb
import fiona
from fiona.crs import from_epsg
import shutil
from tqdm import tqdm
import glob
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Compression
import json
import shapely
from pyproj import Transformer

def merge_masks(files, output_file, tmp_dir, target_crs='EPSG:3857', bounds=None):
    """Merge shapefiles using Fiona for potentially faster execution."""
    # Initialize variables for tracking schemas and writing files
    output_shp = None
    # files = glob.glob(input_dir + '/*/*.shp')
    if os.path.exists(output_file):
        return

    # datasets = [rasterio.open(x) for x in files]
    datasets = []
    for i, file in enumerate(tqdm(files, f'reprojecting files into {target_crs}')):
        src = rasterio.open(file, 'r')
        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src.width, src.height, *src.bounds
        )

        new_metas = src.meta.copy()
        new_metas.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': 255, # avoid 0 be taken as the nodata value
            'compress': 'deflate'
        })

        with rasterio.open(os.path.join(tmp_dir, f'{i}.tif'), 'w', **new_metas) as dst:
            for j in range(1, src.count + 1):
                # Reproject each band
                reproject(
                    source=rasterio.band(src, j),
                    destination=rasterio.band(dst, j),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )

    datasets = [rasterio.open(os.path.join(tmp_dir, f'{idx}.tif'), 'r') for idx in range(len(files))]

    print(f'mosaicing for {output_file}')
    merged_dataset, merged_transform = merge(datasets, nodata=255, bounds=bounds)
    merged_dataset = merged_dataset[0]

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    with rasterio.open(
        os.path.join(output_file, 'mask.tif'), 'w', driver='GTiff',
        height=merged_dataset.shape[0], width=merged_dataset.shape[1], count=1,
        dtype=str(merged_dataset.dtype), crs=datasets[0].crs, transform=merged_transform,
        compress='deflate'
    ) as dst:
        dst.write(merged_dataset, 1)

    os.system(f'rm {tmp_dir}/*')

    with open(os.path.join(output_file, 'finished.txt'), 'w'):
        pass

continents = [
    # 'southamerica/mosaic',
    # 'africa/mosaic',
    # 'africa',
    'asiaeast',
    # 'asiawest',
    # 'europe', 'nordamerica', 'oceania', 'southamerica'
]
# pattern = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/{}/glcv103_guf_wsf'
pattern = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/mask/{}/mosaic'
paths = [pattern.format(x) for x in continents]
out_dir = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/merged_masks'
tmp_dir = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/tmp'
geojson_5x5_dir = '/home/fahong/Code/script/planet_roi_search/guf_geojson/'

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

dir_5x5_paths = {}

for path in paths:
    # path = '/home/Datasets/Dataset4EO/GlobalBF/ai4eo/SOUTHAMERICA/glcv103_guf_wsf'
    for dir_5x5 in tqdm(os.listdir(path), desc=f'searching 5x5 tiles for {path}...'):


        out_path = os.path.join(out_dir, dir_5x5)
        if not os.path.exists(out_path):
        # or not os.path.exists(os.path.join(out_path, 'finished.txt')):
            path_5x5 = os.path.join(path, dir_5x5)
            if not dir_5x5 in dir_5x5_paths:
                dir_5x5_paths[dir_5x5] = []
            else:
                print('path: {path}, dir_5x5: {dir_5x5}')
            dir_5x5_paths[dir_5x5].extend(glob.glob(path_5x5 + '/*/*/*.tif'))


        # out_path = os.path.join(path_5x5, 'merged.shp')
        # merge_shapefiles(path_5x5, out_path)

for key, paths in dir_5x5_paths.items():
    out_path = os.path.join(out_dir, key)
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    cur_json = json.load(open(os.path.join(geojson_5x5_dir, key+'.geojson')))
    poly = shapely.geometry.shape(cur_json['features'][0]['geometry'])
    new_coords = [list(transformer.transform(x, y)) for x, y in poly.exterior.coords]
    poly = shapely.geometry.Polygon(new_coords)

    merge_masks(paths, out_path, tmp_dir, bounds=poly.bounds)
