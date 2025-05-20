import os
import rasterio
import cv2
import numpy as np
import random
from PIL import Image
from multiprocessing.dummy import Pool, Lock
import time
import glob
import json
import csv

ALL_BANDS_S2_L2A = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}

class_sets = {
    19: [
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of"
        " natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    ],
    43: [
        "Agro-forestry areas",
        "Airports",
        "Annual crops associated with permanent crops",
        "Bare rock",
        "Beaches, dunes, sands",
        "Broad-leaved forest",
        "Burnt areas",
        "Coastal lagoons",
        "Complex cultivation patterns",
        "Coniferous forest",
        "Construction sites",
        "Continuous urban fabric",
        "Discontinuous urban fabric",
        "Dump sites",
        "Estuaries",
        "Fruit trees and berry plantations",
        "Green urban areas",
        "Industrial or commercial units",
        "Inland marshes",
        "Intertidal flats",
        "Land principally occupied by agriculture, with significant areas of"
        " natural vegetation",
        "Mineral extraction sites",
        "Mixed forest",
        "Moors and heathland",
        "Natural grassland",
        "Non-irrigated arable land",
        "Olive groves",
        "Pastures",
        "Peatbogs",
        "Permanently irrigated land",
        "Port areas",
        "Rice fields",
        "Road and rail networks and associated land",
        "Salines",
        "Salt marshes",
        "Sclerophyllous vegetation",
        "Sea and ocean",
        "Sparsely vegetated areas",
        "Sport and leisure facilities",
        "Transitional woodland/shrub",
        "Vineyards",
        "Water bodies",
        "Water courses",
    ],
}

label_converter = {
    0: 0,
    1: 0,
    2: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 3,
    15: 3,
    16: 3,
    18: 3,
    17: 4,
    19: 5,
    20: 6,
    21: 7,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    31: 11,
    26: 12,
    27: 12,
    28: 13,
    29: 14,
    33: 15,
    34: 15,
    35: 16,
    36: 16,
    38: 17,
    39: 17,
    40: 18,
    41: 18,
    42: 18,
}

class2idx = {c: i for i, c in enumerate(class_sets[43])}


def normalize(img,mean,std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def get_array(root_dir, patch_id, RGB=False, norm=False):
    data_root_patch = os.path.join(root_dir, patch_id)

    bands = ALL_BANDS_S2_L2A if RGB==False else RGB_BANDS
    MEAN = BAND_STATS['mean']
    STD = BAND_STATS['std']       
    
    chs = []
    for i,band in enumerate(bands):
        patch_path = os.path.join(data_root_patch,f'{patch_id}_{band}.tif')
        with rasterio.open(patch_path) as dataset:
            ch = dataset.read(1)
            ch = cv2.resize(ch, dsize=(120, 120), interpolation=cv2.INTER_LINEAR_EXACT) # [120,120]            
            if norm:
                ch = normalize(ch,mean=MEAN[band],std=STD[band]) # uint8                  
        chs.append(ch)
    img = np.stack(chs, axis=-1) # [264,264,C]

    return img

def get_label(root_dir, patch_id, num_classes=19):
    path = glob.glob(os.path.join(root_dir,patch_id,"*.json"))[0]
    with open(path) as f:
        labels = json.load(f)["labels"]

    indices = [class2idx[label] for label in labels]

    if num_classes == 19:
        indices_optional = [label_converter.get(idx) for idx in indices]
        indices = [idx for idx in indices_optional if idx is not None]

    return indices

class Counter:

    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value

root_dir = 'BigEarthNet-S2-v1.0'
source_dir = 'BigEarthNet-S2-v1.0/BigEarthNet-v1.0'
out_dir = 'BigEarthNet-S2-v1.0/BigEarthNet-v1.0-RGB'
num_workers = 4
counter = Counter()
log_freq = 100

patch_names = os.listdir(source_dir)
indices = range(len(patch_names))

start_time = time.time()
def worker(idx):
    patch_id = patch_names[idx]
    img = get_array(source_dir, patch_id, RGB=True, norm=True)
    img = Image.fromarray(img,mode='RGB')
    img.save(os.path.join(out_dir,patch_id+'.png'))

    label = get_label(source_dir, patch_id)

    with open(root_dir+'/19_labels.csv','a') as wf:
        writer = csv.writer(wf)
        patch_label = label.copy()
        patch_label.insert(0,patch_id)
        writer.writerow(patch_label)


    count = counter.update(1)
    if count % log_freq == 0:
        print(f'Converted {count} images in {time.time() - start_time:.3f}s.')


if num_workers == 0:
    for i in indices:
        worker(i)
else:
    ## parallelism data
    with Pool(processes=num_workers) as p:
        p.map(worker, indices)

