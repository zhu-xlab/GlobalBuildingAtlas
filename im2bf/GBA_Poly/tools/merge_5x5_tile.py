import os
import geopandas as gpd
import pdb
import fiona
from fiona.crs import from_epsg
import shutil
from tqdm import tqdm
import glob

def merge_shapefiles(files, output_file):
    """Merge shapefiles using Fiona for potentially faster execution."""
    # Initialize variables for tracking schemas and writing files
    output_shp = None
    # files = glob.glob(input_dir + '/*/*.shp')
    if os.path.exists(output_file):
        return

    for file_path in tqdm(files, desc=f'processing {output_file}...'):
        # file_path = os.path.join(root, file)
        with fiona.open(file_path, 'r') as input_shp:
            # For the first file, copy the schema and initialize the output file
            if output_shp is None:
                schema = input_shp.schema.copy()
                crs = input_shp.crs
                output_shp = fiona.open(output_file, 'w', driver='ESRI Shapefile', crs=crs, schema=schema)

            # Write records from each input file to the output file
            for record in input_shp:
                output_shp.write(record)

    if output_shp:
        output_shp.close()

    if not os.path.exists(output_file):
        # empty 5x5 tile
        os.makedirs(output_file)

    with open(os.path.join(output_file, 'finished.txt'), 'w'):
        pass

continents = [
    # 'AFRICA/glcv103_guf_wsf', 'ASIAEAST/glcv103_guf_wsf', 'ASIAWEST/glcv103_guf_wsf',
    # 'EUROPE/glcv103_guf_wsf', 'NORDAMERICA/glcv103_guf_wsf', 'OCEANIA/glcv103_guf_wsf',
    # 'SOUTHAMERICA/glcv103_guf_wsf',
    # 'ISLANDS/AFRICA', 'ISLANDS/ASIAEAST', 'ISLANDS/EUROPE', 'ISLANDS/NORDAMERICA',
    # 'ISLANDS/OCEANIA', 'ISLANDS/SOUTHAMERICA'
]
# pattern = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/{}/glcv103_guf_wsf'
# pattern = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/polygon_outputs/{}'
# paths = [pattern.format(x) for x in continents]
# out_dir = '/home/Datasets/ai4eo/Dataset4EO/GlobalBF/merged_polygons'

pattern = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/polygons/'
# paths = [pattern.format(x) for x in continents]
paths = [pattern]
out_dir = '/home/fahong/Datasets/ai4eo/Dataset4EO/GlobalBFV2/merged_polygons'

dir_5x5_paths = {}

for path in paths:
    # path = '/home/Datasets/Dataset4EO/GlobalBF/ai4eo/SOUTHAMERICA/glcv103_guf_wsf'
    for dir_5x5 in tqdm(os.listdir(path), desc=f'searching 5x5 tiles for {path}...'):

        out_path = os.path.join(out_dir, dir_5x5)
        if not os.path.exists(out_path) or not os.path.exists(os.path.join(out_path, 'finished.txt')):
            path_5x5 = os.path.join(path, dir_5x5)
            if not dir_5x5 in dir_5x5_paths:
                dir_5x5_paths[dir_5x5] = []
            else:
                print('path: {path}, dir_5x5: {dir_5x5}')
            dir_5x5_paths[dir_5x5].extend(glob.glob(path_5x5 + '/*/*/*.shp'))


        # out_path = os.path.join(path_5x5, 'merged.shp')
        # merge_shapefiles(path_5x5, out_path)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for key, paths in dir_5x5_paths.items():
    out_path = os.path.join(out_dir, key)
    merge_shapefiles(paths, out_path)
