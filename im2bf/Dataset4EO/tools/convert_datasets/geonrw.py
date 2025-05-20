# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib
from tqdm import tqdm
import glob
import pdb

classes = [
    "forest",
    "water",
    "agricultural",
    "residential,commercial,industrial",
    "grassland,swamp,shrubbery",
    "railway,trainstation",
    "highway,squares",
    "airport,shipyard",
    "roads",
    "buildings"
]

lcov_cmap = matplotlib.colors.ListedColormap(
    [
        "#2ca02c",  # matplotlib green for forest
        "#1f77b4",  # matplotlib blue for water
        "#8c564b",  # matplotlib brown for agricultural
        "#7f7f7f",  # matplotlib gray residential_commercial_industrial
        "#bcbd22",  # matplotlib olive for grassland_swamp_shrubbery
        "#ff7f0e",  # matplotlib orange for railway_trainstation
        "#9467bd",  # matplotlib purple for highway_squares
        "#17becf",  # matplotlib cyan for airport_shipyard
        "#d62728",  # matplotlib red for roads
        "#e377c2",  # matplotlib pink for buildings
    ]
)

train_list = [
    "aachen",
    "bergisch",
    "bielefeld",
    "bochum",
    "bonn",
    "borken",
    "bottrop",
    "coesfeld",
    "dortmund",
    "dueren",
    "duisburg",
    "ennepetal",
    "erftstadt",
    "essen",
    "euskirchen",
    "gelsenkirchen",
    "guetersloh",
    "hagen",
    "hamm",
    "heinsberg",
    "herford",
    "hoexter",
    "kleve",
    "koeln",
    "krefeld",
    "leverkusen",
    "lippetal",
    "lippstadt",
    "lotte",
    "moenchengladbach",
    "moers",
    "muelheim",
    "muenster",
    "oberhausen",
    "paderborn",
    "recklinghausen",
    "remscheid",
    "siegen",
    "solingen",
    "wuppertal",
]

test_list = [
    "duesseldorf",
    "herne",
    "neuss"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GEONRW to mmsegmentation format')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    img_dir = os.path.join(args.out_dir,'img_dir')
    dem_dir = os.path.join(args.out_dir,'dem_dir')
    ann_dir = os.path.join(args.out_dir,'ann_dir')
    
    os.makedirs(img_dir+'/train',exist_ok=True)
    os.makedirs(img_dir+'/test',exist_ok=True)
    os.makedirs(dem_dir+'/train',exist_ok=True)
    os.makedirs(dem_dir+'/test',exist_ok=True)
    os.makedirs(ann_dir+'/train',exist_ok=True)
    os.makedirs(ann_dir+'/test',exist_ok=True)    
    
    for city_name in tqdm(train_list):
        for fname in glob.glob(os.path.join(args.data_path,city_name,'*')):
            if fname.endswith('rgb.jp2'):
                shutil.copy2(fname,img_dir+'/train')
            elif fname.endswith('dem.tif'):
                shutil.copy2(fname,dem_dir+'/train')
            elif fname.endswith('seg.tif'):
                shutil.copy2(fname,ann_dir+'/train')
                
    for city_name in tqdm(test_list):
        for fname in glob.glob(os.path.join(args.data_path,city_name,'*')):
            if fname.endswith('rgb.jp2'):
                shutil.copy2(fname,img_dir+'/test')
            elif fname.endswith('dem.tif'):
                shutil.copy2(fname,dem_dir+'/test')
            elif fname.endswith('seg.tif'):
                shutil.copy2(fname,ann_dir+'/test')


if __name__ == '__main__':
    main()
