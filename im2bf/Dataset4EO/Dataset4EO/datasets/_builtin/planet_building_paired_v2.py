import os
import tarfile
import enum
import functools
import pathlib
from tqdm import tqdm
import h5py
import torch
from typing import Any, Dict, List, Optional, Tuple, BinaryIO, cast, Union
from xml.etree import ElementTree
from Dataset4EO import transforms
import pdb
import numpy as np
import math
from ..utils import clip_big_image
import geopandas as gpd
import rasterio
import shapely.geometry as shgeo
import shapely
import cv2
import json
from pycocotools.coco import COCO as COCO
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
import glob
import itertools
import shutil


from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader, Concater
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    Demultiplexer,
    IterKeyZipper,
    LineReader,
    Zipper,
    IterableWrapper,
    ZipperLongest
)

from torchdata.datapipes.map import SequenceWrapper

from Dataset4EO.datasets.utils import OnlineResource, HttpResource, Dataset, ManualDownloadResource
from Dataset4EO.datasets.utils._internal import (
    path_accessor,
    getitem,
    INFINITE_BUFFER_SIZE,
    path_comparator,
    hint_sharding,
    hint_shuffling,
    read_categories_file,
)
from Dataset4EO.features import BoundingBox, Label, EncodedImage

from .._api import register_dataset, register_info

NAME = "planet_building_paired_v2"
# _TRAIN_LEN_4BAND = 116310
# _TRAIN_LEN_4BAND = 116610
_TRAIN_LEN_4BAND = 135050
# _TEST_LEN_4BAND = 11415
_TEST_LEN_4BAND = 14972
_TEST_LEN_4BAND_100 = 100

_TRAIN_LEN_8BAND = 116184
_TEST_LEN_8BAND = 14972
_TEST_LEN_8BAND_100 = 100

cities_to_continents = {
    "bangibangui": "Africa",  # Assuming it's a misspelling or variation of Bangui
    "beira": "Africa",
    "beni": "Africa",
    "bouake": "Africa",
    "brussel": "Europe",
    "bulawayo": "Africa",
    "bunia": "Africa",
    "chimoio": "Africa",
    "coban": "Africa",
    "copenhagen": "Europe",
    "cumila": "Asia",  # Assuming a variation or misspelling of Comilla, Bangladesh
    "dhaka": "Asia",
    "farafenni": "Africa",
    "juchitan": "South America",  # Juchitán, Mexico
    "matias": "South America",  # Unable to determine without more context
    "nursultan": "Asia",  # Kazakhstan
    "portoviejo": "South America",
    "quelimane": "Africa",
    "salinacruz": "South America",  # Salina Cruz, Mexico
    "sanpedrosula": "South America",  # Honduras
    "santiagodeveraguas": "South America",  # Panama
    "tete": "Africa",
    "tocoa": "South America",  # Honduras
    "tonala": "South America",  # Assuming Tonala, Mexico
    "toronto": "North America",
    "ulaannbaatar": "Asia",  # Mongolia
    'DAL': 'North America',
    'FAI': 'North America',
    'PHI': 'North America',
    'SAN': 'North America',
    'EDM': 'North America',
    'VAN': 'North America',
    'BEI': 'Asia',
    'GUA': 'Asia',
    'SHA': 'Asia',
    'desert': 'Negative',
    'water_body': 'Negative',
    'forest': 'Negative',
    'mountain': 'Negative',
    'asia': 'Negative',
    'southamerica': 'Negative',
    'africa': 'Negative',
    'navarino': 'Negative',
    'amazon': 'Negative',
    'tibet': 'Negative',
    'africa_negative_1': 'Negative',
    'africa_negative_2': 'Negative',
    'africa_negative_3': 'Negative',
    'europe_negative_1': 'Negative',
    'europe_negative_2': 'Negative',
}

city_max_patches = {
    'toronto': 500,
    'DAL': 500,
    'FAI': 500,
    'PHI': 500,
    'SAN': 500,
    'VAN': 500,
    'EDM': 500,
    'BEI': 500,
    'GUA': 500,
    # 'SHA': 500,
    'desert': 200,
    'asia': 200,
    'africa': 200,
    'southamerica': 200,
    'water_body': 200,
    'forest': 200,
    'mountain': 200,
    'amazon': 200,
    'navarino': 200,
    'tibet':200,
    'africa_negative_1': 300,
    'africa_negative_2': 300,
    'africa_negative_3': 300,
    'europe_negative_1': 300,
    'europe_negative_2': 300,
    'bush_1': 300,
    'bush_2': 300,
    'bush_3': 300,
    'bush_4': 300,
    'bush_5': 300,
    'farm_land_1': 300,
    'farm_land_2': 300,
    'moutain_2': 300,
    'moutain_3': 300,
    'moutain_4': 300,
}

np.random.seed(0)

@register_info(NAME)
def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))

class PlanetBuildingResource(ManualDownloadResource):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("Register on https://project.inria.fr/aerialimagelabeling/ and follow the instructions there.", **kwargs)


@register_dataset(NAME)
class PlanetBuildingPairedV2(Dataset):
    """
    - **homepage**: https://project.inria.fr/aerialimagelabeling/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        collect_keys = ['img', 'seg'],
        raster_dir_name='sr_4bands',
        window_root=None,
        window_to_copy_dirs=None,
        window_overide=False,
        num_bands=8,
        split: str = "full",
        skip_integrity_check: bool = True,
        crop_size =  [256, 256],
        stride: int = [192, 192],
        ignore_shp = False,
        additional_raster_resource_names = [],
        ignore_list_path = None
    ) -> None:

        assert split in ('train', 'test', 'full', 'test_100')

        self.window_root = window_root
        self.window_to_copy_dirs = window_to_copy_dirs
        self.num_bands = num_bands
        self.collect_keys = collect_keys
        self._split = split
        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = self._categories
        self.PALETTE = [[0,0,0], [255,255,255]]
        self.crop_size = crop_size
        self.stride=stride
        self.cat_ids = [1]
        self.cat2label = {1: 1}
        self.ignore_shp = ignore_shp
        self.additional_raster_resource_names = additional_raster_resource_names
        self.window_overide = window_overide
        self.raster_dir_name = raster_dir_name
        self.ignore_list = []
        if ignore_list_path is not None:
            self.ignore_list = []
            with open(ignore_list_path) as f:
                for line in f.readlines():
                    img_path, _, _ = line.strip().split(' ')
                    self.ignore_list.append(img_path)

        # assert self.poly_type in ['microsoft_polygon', 'osm_polygon'], 'Invalid type of the shape file!'

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    def _resources(self) -> List[OnlineResource]:

        shp_resource = PlanetBuildingResource(
            file_name = 'shape_files',
            preprocess = None,
            sha256 = None
        )

        raster_resource = PlanetBuildingResource(
            # file_name = f'sr_{self.num_bands}bands',
            file_name = self.raster_dir_name,
            preprocess = None,
            sha256 = None
        )

        ret_resources = [shp_resource, raster_resource]

        if self.additional_raster_resource_names and len(self.additional_raster_resource_names) > 0:

            resources = []
            for name, _ in self.additional_raster_resource_names:
                cur_resource = PlanetBuildingResource(
                    file_name = name,
                    preprocess = None,
                    sha256 = None
                )
                # resources.append(cur_resource)

                ret_resources.append(cur_resource)

        return ret_resources


    def _classify_dp(self, data):
        path = pathlib.Path(data[0])
        flag = False
        for i, (key, value) in enumerate(_POSTFIX_MAP.items()):
            if path.name.endswith(key):
                flag = True
                return i

        return len(_POSTFIX_MAP)

    def split_images(self, image, windows, window_transforms, out_dir, img_name,
                     window_names=None, window_masks=None, crop_size=None,
                     value_map=None, nodata_value=0):

        used_idxes = []
        if crop_size is None:
            crop_size = self.crop_size

        for img_idx, window in enumerate(tqdm(windows)):
            start_x, start_y = int(window.col_off), int(window.row_off)
            # crop_img_name = img_name + f'_{start_x}x{start_y}'
            crop_img_name = window_names[img_idx] if window_names is not None else img_name + f'_{start_x}x{start_y}'
            crop_img_path = os.path.join(out_dir, crop_img_name + '.tif')

            # # window = from_bounds(start_x, start_y, end_x, end_y, image.transform)
            # window = rasterio.windows.Window(start_x, start_y, end_x - start_x, end_y - start_y)
            img_h, img_w = image.shape

            eps = 1e-8
            # if window.col_off >= -eps and window.row_off >= -eps \
            #    and window.col_off + window.width <= img_w + eps and window.row_off + window.height <= img_h + eps:

            window_data = image.read(window=window, resampling=Resampling.nearest)
            if (window_data == nodata_value).all():
                if window_names is None:
                    print(f'Found a patch with all non-data value: {img_name} at location {(start_x, start_y)}, skip this patch')
                    continue

            if window_data.shape[1:] != tuple(crop_size):
                # raise Exception(f'The cropped image generated by {window} gives an unexpected shape {window_data.shape[1:]}')
                # print(f'The cropped image generated by {window} gives an unexpected shape {window_data.shape[1:]}, resize it')
                print(f'The cropped image generated by {window} gives an unexpected shape {window_data.shape[1:]}')
                # import cv2
                # new_window_data = cv2.resize(np.transpose(window_data, [1,2,0]), crop_size, interpolation=cv2.INTER_NEAREST)
                # new_window_data = new_window_data.astype(window_data.dtype)
                # if len(new_window_data.shape) == 2:
                #     new_window_data = np.expand_dims(new_window_data, 2)
                # window_data = np.transpose(new_window_data, [2,0,1])


            if value_map is not None:
                new_data = np.zeros((window_data.shape), dtype=window_data.dtype)
                for key, value in value_map.items():
                    new_data[window_data == key] = value

                window_data = new_data

            out_meta = image.meta.copy()
            out_meta.update({
                        "driver": "GTiff",
                        "height": window_data.shape[1],
                        "width": window_data.shape[2],
                        "transform": window_transforms[img_idx]
                    })
            with rasterio.open(crop_img_path, "w", **out_meta) as dest:
                dest.write(window_data)

            used_idxes.append(img_idx)

            # else:
            #     raise Exception(f'The given window {window} is out of the bound of {img_name}, which is {image.shape} when clipping the image!')

        return used_idxes

    def split_features(self, image, gdf, windows, out_dir, ann_name, transform, window_names=None):

        def get_within_feature_ids(crop_box, bounds, type='has_intersection'):
            """
            type could be has_intersection or contain
            """

            start_x, start_y, end_x, end_y = crop_box
            if type == 'contain':
                flag1 = bounds[:,0] >= start_x
                flag2 = bounds[:,2] < end_x
                flag3 = bounds[:,1] >= start_y
                flag4 = bounds[:,3] < end_y

                return flag1 & flag2 & flag3 & flag4
            else:
                flag1 = bounds[:,0] >= end_x
                flag2 = bounds[:,2] <= start_x
                flag3 = bounds[:,1] >= end_y
                flag4 = bounds[:,3] <= start_y

                return (~(flag1 | flag2)) & (~(flag3 | flag4))

        def has_intersect(points, bbox):
            flag1 = (points[:,0] >= start_x) & (points[:,0] < end_x)
            flag2 = (points[:,1] >= start_y) &(points[:,1] < end_y)
            return (flag1 & flag2).any()

        # polygons = []
        bounds = []
        geo_jsons = []
        for index, row in tqdm(gdf.iterrows(), desc='processing the .shp file'):
            geometry = row['geometry']
            if geometry is not None and geometry.geom_type == 'Polygon':
                cur_bound = np.array(geometry.bounds).reshape(2, 2)
                cur_pixel_bound = np.array([~transform * (x, y) for x, y in cur_bound]).reshape(4,)

                geo_json = shgeo.mapping(geometry)
                pixel_rings = []
                for ring in geo_json['coordinates']:
                    pixel_ring = [~transform * (x, y) for x, y in ring]
                    pixel_rings.append(pixel_ring)

                geo_json['coordinates'] = pixel_rings

                bounds.append(cur_pixel_bound)
                geo_jsons.append(geo_json)

        bounds = np.stack(bounds, axis=0)
        if transform[0] < 0:
            bounds[:, [0,2]] = bounds[:, [2,0]]

        if transform[4] < 0:
            bounds[:, [1,3]] = bounds[:, [3,1]]
        # bounds = bounds[:,[0,3,2,1]]

        ann_id = 0
        # annotations = []
        feature_list = []
        img_h, img_w = image.shape
        for img_id, window in enumerate(tqdm(windows, desc='writing annotations into disk for each bounding box...')):

            start_x, start_y = window.col_off, window.row_off
            end_x, end_y = start_x + window.width, start_y + window.height

            eps = 1e-8
            if window.col_off >= -eps and window.row_off >= -eps \
               and window.col_off + window.width <= img_w + eps and window.row_off + window.height <= img_h + eps:

                crop_box = (start_x, start_y, end_x, end_y)

                ids = get_within_feature_ids(crop_box, bounds, type='has_intersection')
                ids = list(np.where(ids)[0])
                cur_features = [self.add_offset_to_features(geo_jsons[idx], (start_x, start_y)) for idx in ids]

                crop_ann_name = ann_name + f'_{int(start_x)}x{int(start_y)}' if window_names is None else window_names[img_id]
                crop_ann_path = os.path.join(out_dir, crop_ann_name + '.geojson')

                with open(crop_ann_path, 'w') as f:
                    if len(cur_features) > 0:
                        json.dump(cur_features, f)
            else:
                raise Exception(f'The given window {window} is out of the bound of {img_name} when clipping the annotations')

            """
            ## saved cropped image
            for idx in ids:
                minx, miny, maxx, maxy = bounds[idx].tolist()
                w, h = maxx-minx, maxy-miny
                bbox = [minx + w//2 - start_x, miny + h//2 - start_y, w, h]
                annotation = {
                    "id": ann_id,
                    "category_id": 1,  # Building
                    "bbox": [round(x, 2) for x in bbox],
                    # "segmentation": self.sub_polygon(polygon, start_x, start_y)[:-1],
                    "features": self.add_offset_to_features(features[idx], (start_x, start_y)),
                    "score": 1.0,
                    "image_id": img_id,
                    # "area": shgeo.Polygon(polygon).area
                }
                annotations.append(annotation)
                ann_id += 1
            """
    def _filter_shp(self, data):
        return data[0].endswith('.shp')

    def _filter_udm(self, data):
        return not data[0].endswith('_udm.tif')

    def _filter_city(self, data, city_list=None):
        city_name = data.split('/')[-1].split('.')[0]
        for city in city_list:
            if city_name.startswith(city):
                return True
        return False

    def _get_path(self, data):
        return data[0]

    def _key_fn(self, data):
        key = data.split('/')[-1].split('.')[0]
        return key

    def _classify_fn(self, data):
        if data[0].endswith('.tif'):
            return 0
        elif data[0].endswith('.geojson'):
            return 1
        else:
            return 2

    def _parse_dp(self, data, city_name):

        name_map = {
            '4bands_img': 'img',
            '8bands_img': 'img',
        }

        results = dict(
            city_name=city_name,
            continent_name=cities_to_continents[city_name] if city_name in cities_to_continents else None
        )

        for i, key in enumerate(self.collect_keys):
            if key in name_map.keys():
                key = name_map[key]
            if data[i] is not None:
                results[key + '_path'] = data[i][0]
            else:
                results[key + '_path'] = None

        return results

    def _filter_ignore(self, data):
        if data['img_path'] in self.ignore_list:
            return False
        else:
            return True

    def _datapipe(self, resource_dps: List[IterDataPipe]) -> IterDataPipe[Dict[str, Any]]:

        with open(os.path.join(self.root, 'train.txt')) as f:
            temp = f.readlines()
            train_list = [x.strip() for x in temp]

        with open(os.path.join(self.root, 'test.txt')) as f:
            temp = f.readlines()
            test_list = [x.strip() for x in temp]

        city_list = eval(f'{self._split}_list') if self._split in ['train', 'test'] else train_list + test_list

        (shp_dp, raster_dp, *additional_raster_dps) = resource_dps

        if not self.ignore_shp:
            shp_dp = Filter(shp_dp, self._filter_shp)
            shp_dp = Mapper(shp_dp, self._get_path)
            shapes = list(shp_dp)
            shape_list = [x.split('/')[-1].split('.')[0] for x in shapes]

        raster_dp = Filter(raster_dp, self._filter_udm)
        raster_dp = Mapper(raster_dp, self._get_path)
        rasters = list(raster_dp)
        raster_list = [x.split('/')[-1].split('.')[0] for x in rasters]

        common_list = list(set(raster_list).intersection(city_list))
        if not self.ignore_shp:
            common_list = list(set(common_list).intersection(set(shape_list)))

        if len(additional_raster_dps) > 0:
            additional_paths = []
            for cur_raster_dp in additional_raster_dps:
                cur_dp = Filter(cur_raster_dp, self._filter_udm)
                cur_dp = Mapper(cur_dp, self._get_path)
                cur_paths = list(cur_dp)
                cur_city_list = [x.split('/')[-1].split('.')[0] for x in cur_paths]
                common_list = list(set(common_list).intersection(set(cur_city_list)))
                additional_paths.append(cur_paths)

        # if self.use_ndsm_resource:
        #     common_list = list(set(common_list).intersection(set(ndsm_list)))

        if not self.ignore_shp:
            shp_dp = Filter(IterableWrapper(shapes), functools.partial(self._filter_city, city_list=common_list))
        else:
            # create a fake one
            shp_dp = Filter(IterableWrapper(rasters), functools.partial(self._filter_city, city_list=common_list))

        if len(additional_raster_dps) == 0:
            # create a fake one
            additional_raster_dps = Filter(IterableWrapper(rasters), functools.partial(self._filter_city, city_list=common_list))
        else:
            additional_raster_dps = [Filter(
                IterableWrapper(paths), functools.partial(self._filter_city, city_list=common_list)
            ) for paths in additional_paths]
            additional_raster_dps = Zipper(*additional_raster_dps)

        raster_dp = Filter(IterableWrapper(rasters), functools.partial(self._filter_city, city_list=common_list))

        all_data_infos = []
        crop_size, stride = self.crop_size, self.stride
        city_dps = []

        for i, (raster_path, shp_path, additional_raster_paths) in enumerate(Zipper(raster_dp, shp_dp, additional_raster_dps)):
            city_name = raster_path.split('/')[-1].split('.')[0]

            if self.window_root is None:
                clip_dir = os.path.join(self.root, 'clipped', 'c{}_s{}'.format(crop_size[0], stride[0]), city_name)
            else:
                clip_dir = os.path.join(self.root, 'clipped', f'predefined_windows_{self._split}', city_name)

            clip_dir_img = os.path.join(clip_dir, f'{self.num_bands}bands_img')
            clip_dir_ann = os.path.join(clip_dir, f'{self.num_bands}bands_ann')


            os.makedirs(clip_dir, exist_ok=True)
            img_paths = []

            img_tif = rasterio.open(raster_path)
            H, W = img_tif.shape
            # self.geo_transforms[city_name] = img_tif.transform
            geo_transform = img_tif.transform

            # Get the boxes to crop the images, masks, or polygons
            # crop_boxes = self.get_crop_boxes(H, W, crop_size, stride)
            windows = None
            window_names = None
            random_idx = None

            # split images according to the boxes (if not exists)
            if not os.path.exists(clip_dir_img):
                print(f'clipping the images for {city_name}...')
                if self.window_root is not None:
                    if windows is None:
                        windows, window_transforms, window_names = self.get_crop_boxes_by_tifs(self.window_root, city_name, img_tif)

                        if len(windows) == 0:
                            # no windows for this city
                            new_crop_size = []
                            new_crop_size.append(H if crop_size[0] > H else crop_size[0])
                            new_crop_size.append(W if crop_size[1] > W else crop_size[1])
                            new_stride = []
                            new_stride.append(H if crop_size[0] > H else stride[0])
                            new_stride.append(W if crop_size[1] > W else stride[1])
                            crop_boxes = self.get_crop_boxes(H, W, new_crop_size, new_stride)
                            windows = [rasterio.windows.Window(start_x, start_y, end_x - start_x, end_y - start_y) for (start_x, start_y, end_x, end_y) in crop_boxes]
                            window_transforms = [self.calculate_window_geo_transform(geo_transform, window) for window in crop_boxes]
                            window_names = None
                            used_idxes = list(range(len(windows)))

                        if len(windows) > 0:
                            if city_name in city_max_patches.keys():
                                random_idx = np.random.permutation(len(windows))[:city_max_patches[city_name]]
                                windows = [windows[x] for x in random_idx]
                                window_transforms = [window_transforms[x] for x in random_idx]
                                if window_names is not None:
                                    window_names = [window_names[x] for x in random_idx]

                            os.makedirs(clip_dir_img)
                            used_idxes = self.split_images(
                                image=img_tif,
                                windows=windows,
                                window_transforms=window_transforms,
                                window_names=window_names,
                                out_dir=clip_dir_img,
                                img_name=city_name,
                            )
                else:

                    new_crop_size = []
                    new_crop_size.append(H if crop_size[0] > H else crop_size[0])
                    new_crop_size.append(W if crop_size[1] > W else crop_size[1])

                    new_stride = []
                    new_stride.append(H if crop_size[0] > H else stride[0])
                    new_stride.append(W if crop_size[1] > W else stride[1])

                    crop_boxes = self.get_crop_boxes(H, W, new_crop_size, new_stride)
                    # start_x, start_y, end_x, end_y = crop_boxes
                    # windows = [from_bounds(start_x, start_y, end_x, end_y, geo_transform) for (start_x, start_y, end_x, end_y) in crop_boxes]
                    windows = [rasterio.windows.Window(start_x, start_y, end_x - start_x, end_y - start_y) for (start_x, start_y, end_x, end_y) in crop_boxes]
                    window_transforms = [self.calculate_window_geo_transform(geo_transform, window) for window in crop_boxes]

                    if city_name in city_max_patches.keys():
                        random_idx = np.random.permutation(len(windows))[:city_max_patches[city_name]]
                        windows = [windows[x] for x in random_idx]
                        window_transforms = [window_transforms[x] for x in random_idx]

                    os.makedirs(clip_dir_img)
                    # windows, window_transforms, window_names 
                    used_idxes = self.split_images(
                        image=img_tif,
                        windows=windows,
                        window_transforms=window_transforms,
                        window_names=None,
                        out_dir=clip_dir_img,
                        img_name=city_name,
                        crop_size=new_crop_size
                    )
                    # used_idxes = list(range(len(windows)))

            # split ndsms according to the boxes (if not exists)
            if self.additional_raster_resource_names:
                for i, add_raster_path in enumerate(additional_raster_paths):
                    cur_name, cur_map = self.additional_raster_resource_names[i]
                    cur_clip_dir = os.path.join(clip_dir, cur_name)

                    if not os.path.exists(cur_clip_dir):
                        print(f'clipping {self.additional_raster_resource_names[i]} for {city_name}...')
                        cur_raster = rasterio.open(add_raster_path)

                        assert used_idxes is not None, 'when providing additional raster images, a base one should be provided'

                        if self.window_root is not None:
                            new_windows, new_window_transforms, new_window_names = self.get_crop_boxes_by_tifs(
                                self.window_root, city_name, cur_raster
                            )
                            if len(new_windows) > 0:
                                # if windows are provided for this city, replace the windows
                                windows, window_transforms, window_names = new_windows, new_window_transforms, new_window_names
                            else:

                                # no windows for this city
                                new_crop_size = []
                                new_crop_size.append(H if crop_size[0] > H else crop_size[0])
                                new_crop_size.append(W if crop_size[1] > W else crop_size[1])
                                new_stride = []
                                new_stride.append(H if crop_size[0] > H else stride[0])
                                new_stride.append(W if crop_size[1] > W else stride[1])
                                crop_boxes = self.get_crop_boxes(H, W, new_crop_size, new_stride)
                                windows = [rasterio.windows.Window(start_x, start_y, end_x - start_x, end_y - start_y) for (start_x, start_y, end_x, end_y) in crop_boxes]
                                window_transforms = [self.calculate_window_geo_transform(cur_raster.transform, window) for window in crop_boxes]
                                window_names = None

                            if random_idx is not None:
                                windows = [windows[x] for x in random_idx]
                                window_transforms = [window_transforms[x] for x in random_idx]
                                if window_names is not None:
                                    window_names = [window_names[x] for x in random_idx]

                        windows = [windows[x] for x in used_idxes]
                        window_transforms = [window_transforms[x] for x in used_idxes]
                        if window_names is not None:
                            window_names = [window_names[x] for x in used_idxes]

                        os.makedirs(cur_clip_dir)
                        self.split_images(
                            image=cur_raster,
                            windows=windows,
                            window_transforms=window_transforms,
                            window_names=window_names,
                            out_dir=cur_clip_dir,
                            img_name=city_name,
                            value_map=cur_map,
                            nodata_value=-1e9
                        )

            if not os.path.exists(clip_dir_ann) and not self.ignore_shp:
                print(f'clipping the shape file for {city_name}...')
                if self.window_root is not None:
                    if windows is None:
                        windows, window_transforms, window_names = self.get_crop_boxes_by_tifs(self.window_root, city_name, img_tif)

                    gdf = gpd.read_file(shp_path)
                    if gdf.crs.to_epsg() != img_tif.crs:
                        gdf = gdf.to_crs(img_tif.crs)

                    os.makedirs(clip_dir_ann)
                    self.split_features(
                        image=img_tif,
                        gdf=gdf,
                        windows=windows,
                        out_dir=clip_dir_ann,
                        ann_name=city_name,
                        transform=img_tif.transform,
                        window_names=window_names
                    )

            if self.window_to_copy_dirs is not None and len(self.window_to_copy_dirs) > 0:
                assert self.window_root is not None, 'window_root must be provided if you want to copy data according to a list of predefined windows'
                for src_copy_dir in self.window_to_copy_dirs:
                    value_map = None
                    if type(src_copy_dir) == list:
                        # might contain a mapping function
                        src_copy_dir, value_map = src_copy_dir

                    copy_name = src_copy_dir.split('/')[-1]
                    trg_copy_dir = os.path.join(clip_dir, f'{copy_name}')

                    overide = self.window_overide
                    # overide = False
                    if overide or not os.path.exists(trg_copy_dir):
                        to_copy_list = glob.glob(f'{src_copy_dir}/{city_name}*')
                        to_copy_list = np.sort(to_copy_list).tolist()
                        if len(to_copy_list) == 0:
                            continue

                        os.makedirs(trg_copy_dir, exist_ok=overide)
                        print(f'copying dirs for {city_name}...')

                        if random_idx is not None and len(to_copy_list) >= len(random_idx):
                            to_copy_list = [to_copy_list[x] for x in random_idx]

                        for src_path in tqdm(to_copy_list, desc=f'copying from {src_copy_dir} to {trg_copy_dir}'):
                            name = src_path.split('/')[-1]
                            trg_path = os.path.join(trg_copy_dir, name)
                            trg_path = trg_path.replace('.png', '.tif')
                            if value_map:
                                raster = rasterio.open(src_path)
                                data = raster.read()
                                new_data = np.zeros((data.shape), dtype=data.dtype)
                                for key, value in value_map.items():
                                    new_data[data == key] = value

                                out_meta = raster.meta.copy()
                                out_meta.update({
                                            "driver": "GTiff",
                                        })
                                with rasterio.open(trg_path, "w", **out_meta) as dest:
                                    dest.write(new_data)

                            else:
                                # os.system(f'cp {src_path} {trg_path}')
                                shutil.copy(src_path, trg_path)

            collected_dps = []
            for key in self.collect_keys:
                cur_dir = os.path.join(clip_dir, key)
                if not os.path.exists(cur_dir):
                    os.makedirs(cur_dir)

                cur_dp = PlanetBuildingResource(
                    file_name = cur_dir,
                    preprocess = None,
                    sha256 = None
                ).load('')
                collected_dps.append(cur_dp)

            # if len(list(collected_dps[-1])) != len(list(collected_dps[0])):
            #     pdb.set_trace()

            city_dp = ZipperLongest(*collected_dps)
            # city_dp = hint_shuffling(city_dp)
            # city_dp = hint_sharding(city_dp)
            city_dp = Mapper(city_dp, functools.partial(self._parse_dp, city_name=city_name))
            city_dps.append(city_dp)

            # if i == 2:
            #     break

        """
        clip_dir_all_cities = os.path.join(self.root, 'clipped', 'c{}_s{}'.format(crop_size[0], stride[0]))
        clipped_dp = PlanetBuildingResource(
            file_name = clip_dir_all_cities,
            preprocess = None,
            sha256 = None
        ).load('')

        clipped_raster_dp, clipped_shp_dp, _ = Demultiplexer(
            clipped_dp, 3, self._classify_fn, drop_none=True, buffer_size=INFINITE_BUFFER_SIZE
        )
        dp = Zipper(clipped_raster_dp, clipped_shp_dp)
        clipped_dp = Filter(clipped_dp, functools.partial(self._filter_band_num, num_bands=self.num_bands))
        dp = hint_shuffling(dp)
        dp = hint_sharding(dp)
        dp = Mapper(dp, self._parse_dp)
        """
        print(f'Total number of cities: {len(city_dps)}')
        dp = Concater(*city_dps)
        if len(self.ignore_list) > 0:
            print(f'{len(self.ignore_list)} images including the following will be ignored: {self.ignore_list[:5]}')
            dp = Filter(dp, self._filter_ignore)

        if self._split == 'test_100':
            dp = itertools.islice(dp, 100)

        return dp

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def add_offset_to_features(self, features, offset):
        np_offset = np.array(offset).reshape(1,2)
        new_rings = []
        for ring in features['coordinates']:

            new_ring = (np.array(ring) - offset).tolist()
            new_rings.append(new_ring)

        new_features = features.copy()
        new_features['coordinates'] = new_rings
        return new_features

    def get_crop_boxes_by_tifs(self, root, city_name, trg_tif):
        if type(root) == str:
            root = [root]

        tif_list = []
        for cur_root in root:
            tif_list.extend(glob.glob(f'{cur_root}/{city_name}*'))

        tif_list = np.sort(tif_list).tolist()

        windows = []
        window_transforms = []
        window_names = []
        for tif in tif_list:
            try:
                img_tif = rasterio.open(tif)
            except:
                print(tif)
                continue
            bounds = img_tif.bounds
            trg_bounds = transform_bounds(img_tif.crs, trg_tif.crs, *bounds)

            window = from_bounds(*trg_bounds, trg_tif.transform)
            # windows.append(window)
            windows.append(
                Window(
                    col_off=int(round(window.col_off)),
                    row_off=int(round(window.row_off)),
                    width=int(round(window.width)),
                    height=int(round(window.height))
                    # width=self.crop_size[0],
                    # height=self.crop_size[1]
                )
            )
            pixel_bounds = (window.col_off, window.row_off, window.col_off + window.width, window.row_off + window.height)

            # if img_tif.crs != trg_tif.crs:
            #     pdb.set_trace()
            window_transform = self.calculate_window_geo_transform(trg_tif.transform, pixel_bounds)
            window_transforms.append(window_transform)
            window_names.append(tif.split('/')[-1].split('.')[0])

        return windows, window_transforms, window_names


    def get_crop_boxes(self, img_H, img_W, crop_size=(256, 256), stride=(192, 192)):
        # prepare locations to crop

        num_rows = math.ceil((img_H - crop_size[0]) / stride[0]) if \
            math.ceil((img_H - crop_size[0]) / stride[0]) * stride[0] + crop_size[0] >= img_H \
            else math.ceil( (img_H - crop_size[0]) / stride[0]) + 1

        num_cols = math.ceil((img_W - crop_size[1]) / stride[1]) if math.ceil(
            (img_W - crop_size[1]) /
            stride[1]) * stride[1] + crop_size[1] >= img_W else math.ceil(
                (img_W - crop_size[1]) / stride[1]) + 1

        x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
        xmin = x * stride[1]
        ymin = y * stride[0]

        xmin = xmin.ravel()
        ymin = ymin.ravel()
        xmin_offset = np.where(xmin + crop_size[1] > img_W, img_W - xmin - crop_size[1],
                               np.zeros_like(xmin))
        ymin_offset = np.where(ymin + crop_size[0] > img_H, img_H - ymin - crop_size[0],
                               np.zeros_like(ymin))
        boxes = np.stack([
            xmin + xmin_offset, ymin + ymin_offset,
            np.minimum(xmin + crop_size[1], img_W),
            np.minimum(ymin + crop_size[0], img_H)
        ], axis=1)

        return boxes

    """
    def calculate_window_geo_transform(self, original_geo_transform, window):

        (start_x, start_y, end_x, end_y) = window
        ox, oy = original_geo_transform[2], original_geo_transform[5]  # Original top-left coordinates
        px, py = original_geo_transform[0], original_geo_transform[4]  # Pixel width and height
        
        # Calculate new top-left coordinates
        new_ox = ox + (start_x * px)
        new_oy = oy + (start_y * py)
        
        # The rest of the geo-transform remains unchanged
        return rasterio.transform.Affine(new_ox, original_geo_transform[1],
                                         original_geo_transform[2], new_oy,
                                         original_geo_transform[4], original_geo_transform[5])
    """
    def calculate_window_geo_transform(self, original_transform, window):
        start_x, start_y, end_x, end_y = window
        new_c = original_transform.c + original_transform.a * start_x
        new_f = original_transform.f + original_transform.e * start_y

        # Create the new affine transform
        new_transform = rasterio.transform.Affine(original_transform.a, original_transform.b, new_c,
                               original_transform.d, original_transform.e, new_f)

        return new_transform


    def __len__(self) -> int:
        base_len = {
            'train': eval(f'_TRAIN_LEN_{self.num_bands}BAND'),
            'test': eval(f'_TEST_LEN_{self.num_bands}BAND'),
            'test_100': eval(f'_TEST_LEN_{self.num_bands}BAND_100'),
        }[self._split]

        return base_len
