import pandas as pd
import geopandas as gpd
import os
import numpy as np
import fiona
import pyproj
import math
from pyproj import CRS, Transformer
from shapely.geometry import shape, box, Polygon, mapping, MultiPolygon
from fiona import transform as ft
from shapely import wkt
from shapely.ops import transform
from tqdm import tqdm
from glob import glob
from glob import iglob
import shutil

from math import pi, tan, log, cos, inf
# Define functions to convert latitude and longitude to tile coordinates and vice versa

SCHEMA = {
    'geometry': 'Polygon',
    'properties': {
        'source': 'str',
        'id': 'str'
    }
}

SCHEMA_PRODUCT = {
    'geometry': 'Polygon',
    'properties': {
        'source': 'str',
        'id': 'str',
        'region': 'str'
    }
}

device="cpu"

#### Query OSM, MS, and Google Building Footprints.
SCRATCH_DIR = "/dss/lxclscratch/03/ga27lib2"
MS_CACHE_DIR = "/dss/lxclscratch/03/ga27lib2/ga27lib2/ms_cache"
OSM_CACHE_DIR = "/dss/lxclscratch/03/ga27lib2/ga27lib2/osm_cache"
GOOGLE_CACHE_DIR = "/dss/lxclscratch/03/ga27lib2/ga27lib2/google_cache"
GLOBFP_CACHE_DIR = "/dss/lxclscratch/03/ga27lib2/ga27lib2/3dglobfp_cache"
os.makedirs(OSM_CACHE_DIR, exist_ok=True)
os.makedirs(MS_CACHE_DIR, exist_ok=True)
os.makedirs(GOOGLE_CACHE_DIR, exist_ok=True)
os.makedirs(GLOBFP_CACHE_DIR, exist_ok=True)

transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:3857"), always_xy=True)

### Query MS
def sec(x):
    return 1 / cos(x)

def latlon_to_tile(lat, lon, zoom):
    lat_rad = lat * pi / 180
    n = 2 ** zoom
    tileX = int((lon + 180) / 360 * n)
    tileY = int((1 - log(tan(lat_rad) + sec(lat_rad)) / pi) / 2 * n)
    return tileX, tileY

def tile_to_quadkey(tileX, tileY, zoom):
    quadkey = ""
    for i in range(zoom, 0, -1):
        bit = 0
        mask = 1 << (i - 1)
        if tileX & mask:
            bit += 1
        if tileY & mask:
            bit += 2
        quadkey += str(bit)
    return quadkey

# Function to convert ROI bounds to QuadKeys
def roi_bounds_to_quadkeys(latmin, latmax, lonmin, lonmax, zoom_level=9):
    tileX_min, tileY_min = latlon_to_tile(latmin, lonmin, zoom_level)
    tileX_max, tileY_max = latlon_to_tile(latmax, lonmax, zoom_level)
    if tileX_max < tileX_min:
        tileX_min, tileX_max = tileX_max, tileX_min
    if tileY_max < tileY_min:
        tileY_min, tileY_max = tileY_max, tileY_min

    quadkeys = set()
    for tileX in range(tileX_min, tileX_max + 1):
        for tileY in range(tileY_min, tileY_max + 1):
            quadkey = tile_to_quadkey(tileX, tileY, zoom_level)
            quadkeys.add(quadkey)
    
    return quadkeys

def query_3dglobfp_bf(geojson_ours_name):
    print(f"{geojson_ours_name}: Query 3D-GloBFP Building Footprints ...")
    output_path = glob(os.path.join(GLOBFP_CACHE_DIR, geojson_ours_name+"*.geojson"))
    if output_path:
        return output_path[0]
    else:
        return None

def query_ms_bf(bbox):
    print("Query MS Building Footprints")
    latmax_roi, latmin_roi, lonmax_roi, lonmin_roi = bbox
    output_path = os.path.join(MS_CACHE_DIR, f"ms_bf_{latmin_roi}_{latmax_roi}_{lonmin_roi}_{lonmax_roi}.geojson")
    output_flag = os.path.join(MS_CACHE_DIR, f"ms_bf_{latmin_roi}_{latmax_roi}_{lonmin_roi}_{lonmax_roi}.finish")
    if os.path.exists(output_flag):
        return output_path

    bbox_shape = box(lonmin_roi, latmin_roi, lonmax_roi, latmax_roi)
    bbox_in_3857 = transform(transformer.transform, bbox_shape)
    # Read the CSV file
    # df = pd.read_csv("https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv", \
    df = pd.read_csv("microsoft.csv", \
                     dtype={"QuadKey": str})
    # Convert ROI bounds to QuadKeys
    roi_quadkeys = roi_bounds_to_quadkeys(latmin_roi, latmax_roi, lonmin_roi, lonmax_roi)

    # Filter DataFrame based on QuadKeys
    filtered_df = df[df['QuadKey'].isin(roi_quadkeys)]

    file_list = []
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=f"Processing ms building footprints ...:"):
        filename = f"{row['Location']}_{row['QuadKey']}.geojson"
        filepath = os.path.join("microsoft", filename)
        file_list.append(filepath)
    
    if len(file_list) == 0:
        print("No MS BF found!")
        return
    
    feat_shapes = []
    feat_ids = []
    for file_path in tqdm(file_list, desc=f"Reading ms features ...:"):
        with fiona.open(file_path, "r") as features:
            for i, feat in enumerate(features):
                feat_shape = shape(feat.geometry)
                if feat_shape.is_valid:
                    feat_shapes.append(feat_shape)
                    feat_ids.append(feat.properties["id"])

    feat_bounds = np.array([feat.bounds for feat in feat_shapes])
    feat_min_x = feat_bounds[:, 0]
    feat_min_y = feat_bounds[:, 1]
    feat_max_x = feat_bounds[:, 2]
    feat_max_y = feat_bounds[:, 3]
    roi_min_x, roi_min_y, roi_max_x, roi_max_y = bbox_in_3857.bounds
    flag = (roi_max_x < feat_min_x) | (roi_min_x > feat_max_x) | (roi_max_y < feat_min_y) | (roi_min_y > feat_max_y)
    flag = ~flag
    with fiona.open(output_path, "w", crs="EPSG:3857", schema=SCHEMA, driver="GeoJSON") as dst:
        for feat_index in np.nonzero(flag.squeeze())[0]:
            feat_index = int(feat_index)
            feat_geom = feat_shapes[feat_index]
            if feat_geom.intersects(bbox_in_3857):
                new_feat = {
                        "type": "Feature",
                        "geometry": mapping(feat_geom),
                        "properties": {
                            "source": "ms",
                            "id": feat_ids[feat_index]
                        }
                    }
                dst.write(new_feat)
    with open(output_flag, "w") as f:
        f.writelines("finished!")
    return output_path

# def query_ms_bf(geojson_ours_name):
#     print(f"{geojson_ours_name}: Query MS Building Footprints ...")
#     output_path = glob(os.path.join(SCRATCH_DIR, "microsoft", geojson_ours_name+"*.geojson"))
#     if output_path:
#         return output_path[0]
#     else:
#         return None

## Query OSM
# def query_osm_bf(geojson_ours_name):
#     print(f"{geojson_ours_name}: Query OSM Building Footprints ...")
#     output_path = glob(os.path.join(SCRATCH_DIR, "osm", geojson_ours_name+"*.geojson"))
#     if output_path:
#         return output_path[0]
#     else:
#         return None
        
def query_osm_bf(bbox):
    latmax_roi, latmin_roi, lonmax_roi, lonmin_roi = bbox
    gdf = ox.features_from_bbox(bbox=bbox, tags={'building': True})
    gdf = gdf.reset_index()

    gdf = gdf[['osmid', 'geometry']]
    gdf = gdf[gdf.geometry.apply(lambda x: isinstance(x, Polygon))]

    output_filename = f"osm_bf_{latmin_roi}_{latmax_roi}_{lonmin_roi}_{lonmax_roi}.geojson"
    gdf.to_file(os.path.join(OSM_CACHE_DIR, output_filename), driver="GeoJSON")
    return output_filename


### Query Google
def query_google_bf(bbox):
    print("Query Google Building Footprints ...")
    latmax_roi, latmin_roi, lonmax_roi, lonmin_roi = bbox
    output_path = os.path.join(GOOGLE_CACHE_DIR, f"google_bf_{latmin_roi}_{latmax_roi}_{lonmin_roi}_{lonmax_roi}.geojson")
    output_flag = os.path.join(GOOGLE_CACHE_DIR, f"google_bf_{latmin_roi}_{latmax_roi}_{lonmin_roi}_{lonmax_roi}.finish" )
    if os.path.exists(output_flag):
        return output_path
    
    bbox_shape = box(lonmin_roi, latmin_roi, lonmax_roi, latmax_roi)
    bbox_in_3857 = transform(transformer.transform, bbox_shape)

    # Read the GeoJSON tile index file
    # gdf = gpd.read_file("https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson")
    gdf = gpd.read_file(os.path.join(SCRATCH_DIR, "google.geojson"))

    # Filter DataFrame based on bounding box
    filtered_gdf = gdf[gdf.intersects(bbox_shape)]

    file_list = []
    for _, row in filtered_gdf.iterrows():# tqdm(filtered_gdf.iterrows(), total=len(filtered_gdf), desc=f"Processing google building footprints ...:"):
        filename = f"{row['tile_id']}.geojson"
        filepath = os.path.join(SCRATCH_DIR, "google", filename)
        flagpath = filepath.replace(".geojson", ".finish")
        file_list.append(filepath)
    if len(file_list) == 0:
        print("No Google BF found!")
        return
    
    feat_shapes = []
    feat_ids = []
    for file_path in file_list: # tqdm(file_list, desc=f"Reading google features ...:"):
        with fiona.open(file_path, "r") as features:
            for i, feat in enumerate(features):
                feat_shape = shape(feat.geometry)
                if feat_shape.is_valid:
                    feat_shapes.append(feat_shape)
                    feat_ids.append(feat.properties["id"])

    feat_bounds = np.array([feat.bounds for feat in feat_shapes])
    feat_min_x = feat_bounds[:, 0]
    feat_min_y = feat_bounds[:, 1]
    feat_max_x = feat_bounds[:, 2]
    feat_max_y = feat_bounds[:, 3]
    roi_min_x, roi_min_y, roi_max_x, roi_max_y = bbox_in_3857.bounds
    flag = (roi_max_x < feat_min_x) | (roi_min_x > feat_max_x) | (roi_max_y < feat_min_y) | (roi_min_y > feat_max_y)
    flag = ~flag

    with fiona.open(output_path, "w", crs="EPSG:3857", schema=SCHEMA, driver="GeoJSON") as dst:
        for feat_index in np.nonzero(flag.squeeze())[0]:
            feat_index = int(feat_index)
            feat_geom = feat_shapes[feat_index]
            if feat_geom.intersects(bbox_in_3857):
                new_feat = {
                        "type": "Feature",
                        "geometry": mapping(feat_geom),
                        "properties": {
                            "source": "google",
                            "id": feat_ids[feat_index]
                        }
                    }
                dst.write(new_feat)
    with open(output_flag, "w") as f:
        f.writelines("finished!")
    return output_path

#### Evaluate different sources.
### When OSM is not available.
def get_total_area(source_file):
    areas = 0
    with fiona.open(source_file, "r") as feat_src:
        for feat in feat_src:
            geom = shape(feat.geometry)
            area = geom.area
            areas += area
        return areas, list(range(len(feat_src)))

### When OSM is available.
def compute_intersection_over_osm(osm_shapes, source_shapes, osm_bounds, source_bounds, source_name):
    osm_min_x, osm_min_y, osm_max_x, osm_max_y = osm_bounds[:, 0], osm_bounds[:, 1], osm_bounds[:, 2], osm_bounds[:, 3]
    source_min_x, source_min_y, source_max_x, source_max_y = source_bounds[:, 0], source_bounds[:, 1], source_bounds[:, 2], source_bounds[:, 3]
    flag = (osm_max_x[:, np.newaxis] < source_min_x) | (osm_min_x[:, np.newaxis] > source_max_x) | \
           (osm_max_y[:, np.newaxis] < source_min_y) | (osm_min_y[:, np.newaxis] > source_max_y)
    flag = ~flag

    intersecting_features = []
    total_area = 0
    for osm_index in range(len(osm_shapes)):
    # for osm_index in tqdm(range(len(osm_shapes)), desc=f"Checking {source_name} intersection over osm ...:", total=len(osm_shapes)):
        osm_geom = osm_shapes[osm_index]
        for feat_index in np.nonzero(flag[osm_index])[0]:
            feat_index = int(feat_index)
            feat_geom = source_shapes[feat_index]
            if osm_geom.intersects(feat_geom):
                total_area += osm_geom.intersection(feat_geom).area
                intersecting_features.append(feat_index)
    diff_features = set(range(len(source_shapes))) - set(intersecting_features)
    diff_features = sorted(diff_features)
    return total_area, diff_features

def compute_intersection_over_osm_by_chunk(osm_shapes, source_shapes, osm_bounds, source_bounds, source_name):
    if (len(osm_shapes) * len(source_shapes) < 5e9):
        return compute_intersection_over_osm(osm_shapes, source_shapes, osm_bounds, source_bounds, source_name)
    
    chunk_num_each_dim = min(int(math.sqrt(len(osm_shapes) * len(source_shapes) / (1e9))), 10)

    osm_min_x, osm_min_y, osm_max_x, osm_max_y = osm_bounds[:, 0], osm_bounds[:, 1], osm_bounds[:, 2], osm_bounds[:, 3]
    osm_min_xx, osm_min_yy, osm_max_xx, osm_max_yy = osm_min_x.min(), osm_min_y.min(), osm_max_x.max(), osm_max_y.max()

    chunk_xs = np.linspace(osm_min_xx, osm_max_xx, chunk_num_each_dim+1)
    chunk_ys = np.linspace(osm_min_yy, osm_max_yy, chunk_num_each_dim+1)
    chunk_min_x, chunk_max_x = chunk_xs[:-1], chunk_xs[1:]
    chunk_min_y, chunk_max_y = chunk_ys[:-1], chunk_ys[1:]

    chunk_bounds = np.array([
        (xmin, ymin, xmax, ymax)
        for xmin, xmax in zip(chunk_min_x, chunk_max_x)
        for ymin, ymax in zip(chunk_min_y, chunk_max_y)
    ])

    osm_flag = compute_intersection_over_chunk(chunk_bounds, osm_shapes, osm_bounds)
    source_flag = compute_intersection_over_chunk(chunk_bounds, source_shapes, source_bounds)

    total_area = 0
    diff_features = []
    for chunk_ind in range(chunk_num_each_dim * chunk_num_each_dim):
        osm_intersect_feature_indices = np.nonzero(osm_flag[chunk_ind])[0].astype(int).tolist()
        source_intersect_feature_indices = np.nonzero(source_flag[chunk_ind])[0].astype(int).tolist()

        area, diff_feat = compute_intersection_over_osm_single_chunk(osm_shapes, source_shapes, osm_bounds, source_bounds, osm_intersect_feature_indices, source_intersect_feature_indices, source_name)
        total_area += area
        diff_features.extend(diff_feat)
    return total_area, diff_features
    
def compute_intersection_over_chunk(chunk_bounds, source_shapes, source_bounds):
    chunk_min_x, chunk_min_y, chunk_max_x, chunk_max_y = chunk_bounds[:, 0], chunk_bounds[:, 1], chunk_bounds[:, 2], chunk_bounds[:, 3]
    source_min_x, source_min_y, source_max_x, source_max_y = source_bounds[:, 0], source_bounds[:, 1], source_bounds[:, 2], source_bounds[:, 3]
    flag = (chunk_max_x[:, np.newaxis] < source_min_x) | (chunk_min_x[:, np.newaxis] > source_max_x) | \
           (chunk_max_y[:, np.newaxis] < source_min_y) | (chunk_min_y[:, np.newaxis] > source_max_y)
    flag = ~flag
    return flag
            

def compute_intersection_over_osm_single_chunk(osm_shapes, source_shapes, osm_bounds, source_bounds, osm_indices, source_indices, source_name):
    osm_min_x, osm_min_y, osm_max_x, osm_max_y = osm_bounds[osm_indices, 0], osm_bounds[osm_indices, 1], osm_bounds[osm_indices, 2], osm_bounds[osm_indices, 3]
    source_min_x, source_min_y, source_max_x, source_max_y = source_bounds[source_indices, 0], source_bounds[source_indices, 1], source_bounds[source_indices, 2], source_bounds[source_indices, 3]
    flag = (osm_max_x[:, np.newaxis] < source_min_x) | (osm_min_x[:, np.newaxis] > source_max_x) | \
           (osm_max_y[:, np.newaxis] < source_min_y) | (osm_min_y[:, np.newaxis] > source_max_y)
    flag = ~flag
    
    intersecting_features = []
    total_area = 0
    for osm_index in range(len(osm_indices)):
    # for osm_index in tqdm(range(len(osm_shapes)), desc=f"Checking {source_name} intersection over osm single chunk...:", total=len(osm_shapes)):
        osm_geom = osm_shapes[osm_indices[osm_index]]
        for feat_index in np.nonzero(flag[osm_index])[0]:
            feat_index = int(feat_index)
            feat_geom = source_shapes[source_indices[feat_index]]
            if osm_geom.intersects(feat_geom):
                total_area += osm_geom.intersection(feat_geom).area
                intersecting_features.append(source_indices[feat_index])
    diff_features = set(source_indices) - set(intersecting_features)
    diff_features = sorted(diff_features)
    return total_area, diff_features

def compute_difference_from_osm(source_shapes, diff_ind, source_name):
    total_area = 0
    for feat_index in diff_ind: #tqdm(diff_ind, desc=f"Reading {source_name} difference ...:"):
        feat_geom = source_shapes[feat_index]
        total_area += feat_geom.area
    return total_area

#### Aggregation.
### For one district.
def aggregate_osm_and_source(osm_file, best_source, diff_ind, best_source_name, output_file, main_source):
    with fiona.open(output_file, "w", driver="GeoJSON", schema=SCHEMA, crs="EPSG:3857") as feat_output:
        if osm_file:
            with fiona.open(osm_file, "r") as feat_osm:
                for osm in feat_osm:
                    new_feat = {
                        "type": "Feature",
                        "geometry": osm.geometry,
                        "properties": {
                            "source": main_source,
                            "id": osm.properties["id"]
                        }
                    }
                    feat_output.write(new_feat)
        if best_source_name is not None:
            with fiona.open(best_source, "r", driver="GeoJSON") as feat_src:
                for feat_ind in diff_ind:
                    new_feat = {
                        "type": "Feature",
                        "geometry": feat_src[feat_ind].geometry,
                        "properties": {
                            "source": best_source_name,
                            "id": feat_src[feat_ind].properties["id"]
                        }
                    }
                    feat_output.write(new_feat)
    with open(os.path.join(os.path.dirname(output_file), "finished.txt"), "w") as f:
        f.writelines("finished!")

### For one 5x5 tile.
def merge_all_results(result_list, output_file):
    feat_list = []
    with fiona.open(output_file, "w", driver="GeoJSON", schema=SCHEMA_PRODUCT, crs="EPSG:3857") as dst:
        for result in result_list: #tqdm(result_list, desc="Merging all the results ...:"):
            region = result.split("/")[-2].split("-")[0]
            with fiona.open(result, "r") as features:
                for feat in features:
                    # id = feat.properties["source"]+feat.properties["id"]
                    # if feat.properties["source"] in ["ms", "google", "osm"]:
                    #     if id in feat_list:
                    #         continue
                    #     else:
                    #         feat_list.append(id)
                    feat["properties"].update({
                        "region": region
                    })
                    dst.write(feat)


# def clear_all_chunk_results(root_dir):
#     for root, dirs, files in os.walk(root_dir, topdown=False):
#         for name in files:
#             if name.startswith('chunk'):
#                 os.remove(os.path.join(root, name))  # Delete file

#         for name in dirs:
#             if name.startswith('chunk'):
#                 shutil.rmtree(os.path.join(root, name))  # Delete directory and its contents

#     print(f"All files and folders starting with 'chunk' in {root_dir} and its subfolders have been deleted.")

def clear_all_chunk_results(root_dir):
    # Delete all files starting with "chunk"
    for file in iglob(f"{root_dir}/**/chunk*", recursive=True):
        if os.path.isfile(file):
            os.remove(file)
        elif os.path.isdir(file):
            shutil.rmtree(file)

    print(f"All files and folders starting with 'chunk' in {root_dir} and its subfolders have been deleted.")

### Other utils used by main.
def get_bounds_in_4326(geojson_file, in_crs):
    with fiona.open(geojson_file, "r") as f:
        bbox = f.bounds
        bbox = transform_bounds(bbox, in_crs, "EPSG:4326")
    return bbox

def transform_bounds(bbox, in_crs, out_crs):
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True)
    xs, ys = (bbox[0], bbox[2]), (bbox[1], bbox[3])
    xs, ys = transformer.transform(xs, ys)
    return (ys[1], ys[0], xs[1], xs[0])

def split_polygon_cross_180(geometry):
    """Check if a Shapely geometry crosses the 180° meridian."""
    left_parts = []
    right_parts = []
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]

    for poly in polygons:
        left_coords = []
        right_coords = []

        # Process the exterior ring
        for lon, lat in poly.exterior.coords:
            if lon < 0:
                left_coords.append((min(lon + 360, 180), lat))  # Shift left part
            else:
                left_coords.append((lon, lat))  # Keep as is

            if lon > 0:
                right_coords.append((max(lon - 360, -180), lat))  # Shift right part
            else:
                right_coords.append((lon, lat))  # Keep as is

        if left_coords:
            left_parts.append(Polygon(left_coords))
        if right_coords:
            right_parts.append(Polygon(right_coords))
        roi_shape_left = MultiPolygon(left_parts) if len(left_parts) > 1 else left_parts[0] if left_parts else None
        roi_shape_right = MultiPolygon(right_parts) if len(right_parts) > 1 else right_parts[0] if right_parts else None
    return roi_shape_left, roi_shape_right


def get_roi_shapes_and_bounds(roi_geojson, bbox_shape):
    roi_shapes = []
    roi_ids = []
    with fiona.open(roi_geojson, "r") as f:
        for roi in f: 
        # for roi in tqdm(f, desc=f"{os.path.basename(roi_geojson)}: Reading RoI bounds ... :", total=len(f)):
            roi_geom_4326 = roi.geometry
            roi_shape_4326 = shape(roi_geom_4326)
            xmin, ymin, xmax, ymax = roi_shape_4326.bounds
            if xmin < -170 and xmax > 170:
                left_shape, right_shape = split_polygon_cross_180(roi_shape_4326)
                if (not left_shape.intersects(bbox_shape)) and (not right_shape.intersects(bbox_shape)):
                    continue

            roi_geom = ft.transform_geom(f.crs, "EPSG:3857", roi_geom_4326)
            roi_shape = shape(roi_geom)

            if roi_shape.is_valid:
                roi_shapes.append(roi_shape)

                roi_id = "-".join([roi.properties[f"GID_{i}"] for i in range(6)]).rstrip("-")
                roi_ids.append(roi_id)
    roi_bounds = np.array([roi.bounds for roi in roi_shapes])
    return roi_shapes, roi_bounds, roi_ids


def get_source_shapes_and_bounds(source_geojson, source_name):
    feat_shapes = []
    feat_ids = []
    if source_name == "osm":
        id_field = "id"
        with fiona.open(source_geojson, "r") as f:
            for feat in f:
            # for feat in tqdm(f, desc=f"Reading {source_name} features ...:", total=len(f)):
                feat_geom = feat.geometry
                feat_geom = ft.transform_geom(f.crs, "EPSG:3857", feat_geom)
                feat_shape = shape(feat_geom)
                feat_id = feat.properties[id_field]
                if (feat_shape.is_valid): # & (feat_id not in feat_ids):
                    feat_shapes.append(feat_shape)
                    feat_ids.append(feat_id)
    elif source_name == "ours2":
        id_field = "FID"
        with fiona.open(source_geojson, "r") as f:
            for feat in f:
            # for feat in tqdm(f, desc=f"Reading {source_name} features ...:", total=len(f)):
                feat_geom = feat.geometry
                feat_geom = ft.transform_geom(f.crs, "EPSG:3857", feat_geom)
                feat_shape = shape(feat_geom)
                if feat_shape.is_valid:
                    feat_shapes.append(feat_shape)
                    feat_ids.append(feat.properties[id_field])
    else:
        id_field = "id"
        with fiona.open(source_geojson, "r") as f:
            for feat in f:
            # for feat in tqdm(f, desc=f"Reading {source_name} features ...:", total=len(f)):
                feat_geom = feat.geometry
                feat_geom = ft.transform_geom(f.crs, "EPSG:3857", feat_geom)
                feat_shape = shape(feat_geom)
                if feat_shape.is_valid:
                    feat_shapes.append(feat_shape)
                    feat_ids.append(feat.properties[id_field])

    feat_bounds = np.array([feat.bounds for feat in feat_shapes])
    return feat_shapes, feat_bounds, feat_ids

def get_source_shapes_and_bounds_chunked(chunk, chunk_ind, source_name, source_crs):
    feat_shapes = []
    feat_ids = []
    if source_name == "osm":
        id_field = "id"
        for feat in chunk:
        # for feat in tqdm(f, desc=f"Reading {source_name} features ...:", total=len(f)):
            feat_geom = feat.geometry
            feat_geom = ft.transform_geom(source_crs, "EPSG:3857", feat_geom)
            feat_shape = shape(feat_geom)
            feat_id = feat.properties[id_field]
            if (feat_shape.is_valid): # & (feat_id not in feat_ids):
                feat_shapes.append(feat_shape)
                feat_ids.append(feat_id)
    elif source_name == "ours2":
        id_field = "FID"
        for feat in chunk:
        # for feat in tqdm(f, desc=f"Reading {source_name} features ...:", total=len(f)):
            feat_geom = feat.geometry
            feat_geom = ft.transform_geom(source_crs, "EPSG:3857", feat_geom)
            feat_shape = shape(feat_geom)
            if feat_shape.is_valid:
                feat_shapes.append(feat_shape)
                feat_ids.append(feat.properties[id_field])
    else:
        id_field = "id"
        for feat in chunk:
        # for feat in tqdm(f, desc=f"Reading {source_name} features ...:", total=len(f)):
            feat_geom = feat.geometry
            feat_geom = ft.transform_geom(source_crs, "EPSG:3857", feat_geom)
            feat_shape = shape(feat_geom)
            if feat_shape.is_valid:
                feat_shapes.append(feat_shape)
                feat_ids.append(feat.properties[id_field])

    feat_bounds = np.array([feat.bounds for feat in feat_shapes])
    return feat_shapes, feat_bounds, feat_ids
# feat_shapes, feat_bounds, feat_ids = get_source_shapes_and_bounds(geojson_source, name_source)
# logging.info(f"{geojson_ours_name}: {name_source}: Processing {len(feat_ids)} features...")
# if len(feat_ids) > 0:
#     coarse_intersection_flag = get_coarse_intersection_flag(roi_bounds_todo, feat_bounds)
#     logging.info(f"{geojson_ours_name}: {name_source}: Coarse intersection flag matrix {coarse_intersection_flag.shape} got ...")

#     for roi_index in range(len(roi_shapes_todo)): #tqdm(range(len(roi_shapes_todo)), desc=f"Cropping {name_source} into RoIs ...", total=len(roi_shapes_todo)):
#         roi_geom = roi_shapes_todo[roi_index]
#         roi_id = roi_ids_todo[roi_index]
#         output_folder = os.path.join(cropped_file_dir, roi_id)
#         output_flag = os.path.join(output_folder, name_source + ".finish")
#         if os.path.exists(output_flag):
#             continue

#         os.makedirs(output_folder, exist_ok=True)
#         logging.info(f"{geojson_ours_name}: {name_source}: RoI {roi_id}: Processing {coarse_intersection_flag[roi_index].sum()} features...")
        
#         write_cropped_features(feat_shapes, feat_ids, roi_geom, np.nonzero(coarse_intersection_flag[roi_index])[0], name_source, os.path.join(output_folder, name_source + ".geojson"))
                            
def chunked_processing_cropping(chunk, chunk_ind, roi_ids, roi_shapes, roi_bounds, output_folder, name_source, source_crs):
    # output_path = os.path.join(output_dir, ":05d".format(chunk_id), name_source+".geojson")
    chunk_output_flag = os.path.join(output_folder, "chunk_"+name_source+f"_{chunk_ind:05d}.finish")

    feat_shapes, feat_bounds, feat_ids = get_source_shapes_and_bounds_chunked(chunk, chunk_ind, name_source, source_crs)
    coarse_intersection_flag = get_coarse_intersection_flag(roi_bounds, feat_bounds)
    for roi_index in range(len(roi_shapes)):
        roi_geom = roi_shapes[roi_index]
        roi_id = roi_ids[roi_index]
        output_dir = os.path.join(output_folder, roi_id, f"chunk_{chunk_ind:05d}")

        output_flag = os.path.join(output_dir, "chunk_"+name_source+".finish")
        if os.path.exists(output_flag):
            continue

        feature_intersected_indices = np.nonzero(coarse_intersection_flag[roi_index])[0]
        if feature_intersected_indices.size > 0:
            os.makedirs(output_dir, exist_ok=True)
            write_cropped_features(feat_shapes, feat_ids, roi_geom, np.nonzero(coarse_intersection_flag[roi_index])[0], name_source, os.path.join(output_dir, "chunk_"+name_source +".geojson"))
    with open(chunk_output_flag, "w") as f:
        f.writelines("finished!\n")

def get_coarse_intersection_flag(roi_bounds, feat_bounds):
    # Extract bounding box coordinates for ROI and features
    roi_min_x, roi_min_y, roi_max_x, roi_max_y = roi_bounds[:, 0], roi_bounds[:, 1], roi_bounds[:, 2], roi_bounds[:, 3]
    feat_min_x, feat_min_y, feat_max_x, feat_max_y = feat_bounds[:, 0], feat_bounds[:, 1], feat_bounds[:, 2], feat_bounds[:, 3]
    
    # Compute the flag indicating non-intersection using broadcasting
    # Add np.newaxis to align dimensions for broadcasting
    flag = (roi_max_x[:, np.newaxis] < feat_min_x) | (roi_min_x[:, np.newaxis] > feat_max_x) | \
           (roi_max_y[:, np.newaxis] < feat_min_y) | (roi_min_y[:, np.newaxis] > feat_max_y)
    
    # Invert the flag to indicate intersection
    flag = ~flag
    
    return flag

# def write_cropped_features(intersecting_features, output_path):
#     with fiona.open(output_path, "w", driver="GeoJSON", crs="EPSG:3857", schema=SCHEMA) as dst:
#         for feature in intersecting_features:
#             if feature["geometry"]["type"] == "Polygon":
#                 dst.write(feature)
#             elif feature["geometry"]["type"] == "MultiPolygon":
#                 feat_shape = shape(feature["geometry"])
#                 for i, poly in enumerate(feat_shape.geoms):
#                     new_feat = {
#                         "type": "Feature",
#                         "geometry": mapping(poly),
#                         "properties": {
#                             "source": feature["properties"]["source"],
#                             "id": str(feature["properties"]["id"])+"_"+str(i)
#                         }
#                     }
#                     dst.write(new_feat)

#     with open(output_path.replace(".geojson", ".finish"), "w") as f:
#         f.writelines("finished!\n")

def write_cropped_features(feat_shapes, feat_ids, roi_geom, cif_ind, name_source, output_path):
    with fiona.open(output_path, "w", driver="GeoJSON", crs="EPSG:3857", schema=SCHEMA) as dst:
        for feat_index in cif_ind:
            feat_index = int(feat_index.item())
            feat_geom = feat_shapes[feat_index]
            if roi_geom.intersects(feat_geom):
                if feat_geom.geom_type == "Polygon":
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(feat_geom),
                        "properties": {
                            "source": name_source,
                            "id": feat_ids[feat_index],
                        }
                    }
                    dst.write(feature)
                elif feat_geom.geom_type == "MultiPolygon":
                    for i, poly in enumerate(feat_geom.geoms):
                        new_feat = {
                            "type": "Feature",
                            "geometry": mapping(poly),
                            "properties": {
                                "source": name_source,
                                "id": str(feat_ids[feat_index])+"_"+str(i),
                            }
                        }
                        dst.write(new_feat)
    
    with open(output_path.replace(".geojson", ".finish"), "w") as f:
        f.writelines("finished!\n")

def get_best_without_osm(district_folder, main_source):
    max_metric = -inf
    best_source_name = None
    best_source = None
    best_diff = None
    for source in ["ms", "osm", "ours2", "google"]:
        if source == main_source:
            continue
        source_file = os.path.join(district_folder, source+".geojson")
        if os.path.exists(source_file):
            metric, total_ind = get_total_area(source_file)
            if metric > max_metric:
                max_metric = metric
                best_source_name = source
                best_source = source_file
                best_diff = total_ind
    # print(district_folder, best_source_name, len(best_diff))
    return best_source, best_source_name, best_diff

def get_best_with_osm(district_folder, main_source):
    osm_file = os.path.join(district_folder, main_source+".geojson")
    max_metric = -inf
    best_source_name = None
    best_source = None
    best_diff = None
    with fiona.open(osm_file, "r") as feat_osm:
        if len(feat_osm) == 0:
            return get_best_without_osm(district_folder, main_source)

        osm_shapes = []
        osm_properties = []
        for osm in feat_osm:
        # for osm in tqdm(feat_osm, desc=f"Reading osm features ...:", total=len(feat_osm)):
            osm_shape = shape(osm.geometry)
            if osm_shape.is_valid:
                osm_shapes.append(osm_shape)
                osm_properties.append(osm["properties"])
        osm_bounds = np.array([osm_shape.bounds for osm_shape in osm_shapes])

        for source in ["ms", "osm", "ours2", "google"]:
            if source == main_source:
                continue
            source_file = os.path.join(district_folder, source+".geojson")
            if not os.path.exists(source_file):
                continue
            source_shapes = []
            source_properties = []
            with fiona.open(source_file, "r") as feat_source:
                for feat in feat_source:
                # for feat in tqdm(feat_source, desc=f"Reading {source} features ...:", total=len(feat_source)):
                    feat_shape = shape(feat.geometry)
                    if feat_shape.is_valid:
                        source_shapes.append(feat_shape)
                        source_properties.append(feat["properties"])
            source_bounds = np.array([source_shape.bounds for source_shape in source_shapes])
            if len(source_shapes) == 0:
                continue
            # intersection_over_osm, diff_ind = compute_intersection_over_osm(osm_shapes, source_shapes, osm_bounds, source_bounds, source)
            intersection_over_osm, diff_ind = compute_intersection_over_osm_by_chunk(osm_shapes, source_shapes, osm_bounds, source_bounds, source)
            difference_from_osm = compute_difference_from_osm(source_shapes, diff_ind, source)
            metric = intersection_over_osm + difference_from_osm

            if metric > max_metric:
                max_metric = metric
                best_source_name = source
                best_source = source_file
                best_diff = diff_ind
    return best_source, best_source_name, best_diff


def no_crop_copy(geojson_source, cropped_output_file, name_source):
    with fiona.open(geojson_source, "r") as fshp:
        crs = fshp.crs
        shp_schema = fshp.schema
        with fiona.open(cropped_output_file, "w", driver="GeoJSON", crs=crs, schema=SCHEMA) as fgeojson:
            for feat in fshp:
                if "id" in shp_schema["properties"].keys():
                    feat_id = feat["properties"]["id"]
                else:
                    feat_id = feat["properties"]["FID"]
                new_feat = {
                    "type": "Feature",
                    "geometry": feat["geometry"],
                    "properties":{
                        "source": name_source,
                        "id": str(feat_id)
                    }
                }
                fgeojson.write(new_feat)
