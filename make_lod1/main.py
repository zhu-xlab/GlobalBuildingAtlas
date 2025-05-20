import argparse
import rasterio
from rasterio import mask as mk
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box, mapping, shape
import fiona
import numpy as np
from tqdm import tqdm
import os
from glob import glob
import logging
import traceback
import multiprocessing 
from pyproj import Transformer, CRS
import pdb
import time
import re

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
SCHEMA0 = {
    'geometry': 'Polygon',
    'properties': {
        'source': 'str',
        'id': 'str',
        "region": "str"
    }
}

SCHEMA = {
    'geometry': 'Polygon',
    'properties': {
        'source': 'str',
        'id': 'str',
        "height": "float",
        "var": "float",
        "region": "str"
    }
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'

def get_source_shapes_and_bounds(source_geojson):
    feat_sources = []
    feat_shapes = []
    feat_ids = []
    feat_regions = []

    with fiona.open(source_geojson, "r") as f:
        for feat in f:
            feat_geom = feat.geometry
            feat_shape = shape(feat_geom)
            feat_source = feat.properties["source"]
            feat_id = feat.properties["id"]
            feat_region = feat.properties["region"]

            if (feat_shape.is_valid):
                feat_sources.append(feat_source)
                feat_shapes.append(feat_shape)
                feat_ids.append(feat_id)
                feat_regions.append(feat_region)

    feat_bounds = np.array([feat.bounds for feat in feat_shapes])
    return feat_sources, feat_shapes, feat_bounds, feat_ids, feat_regions

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

def write_cropped_features(feat_sources, feat_shapes, feat_ids, feat_regions, roi_geom, cif_ind, output_path):
    with fiona.open(output_path, "w", driver="GeoJSON", crs="EPSG:3857", schema=SCHEMA0) as dst:
        for feat_index in cif_ind:
            feat_index = int(feat_index.item())
            feat_geom = feat_shapes[feat_index]
            if roi_geom.intersects(feat_geom):
                feature = {
                    "type": "Feature",
                    "geometry": mapping(feat_geom),
                    "properties": {
                        "source": feat_sources[feat_index],
                        "id": feat_ids[feat_index],
                        "region": feat_regions[feat_index]
                    }
                }
                dst.write(feature)
    with open(output_path.replace(".geojson", ".finish"), "w") as f:
        f.writelines("finished!\n")

def merge_single_result(input_vector, continent):
    input_base = os.path.basename(input_vector)
    # print(os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/lod1_temp_max", continent, input_base.split(".")[0], "*.geojson"))
    result_list = glob(os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/lod1_temp_max2", continent, input_base.split(".")[0], "*.geojson"))
    output_file = os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/lod1_max2", continent, input_base.split(".")[0]+"_lod1.geojson")
    output_flag = output_file.replace(".geojson", ".finish")

    # feat_list = []
    with fiona.open(output_file, "w", driver="GeoJSON", schema=SCHEMA, crs="EPSG:3857") as dst:
        for result in result_list: #tqdm(result_list, desc="Merging all the results ...:", total=len(result_list)):
            with fiona.open(result, "r") as features:
                for feat in features:
                    dst.write(feat)

    with open(output_flag, "w") as f:
        f.writelines("Finished!\n")

def write_chunk(chunk, output_path, schema, crs):
    with fiona.open(output_path, "w", schema=schema, crs=crs) as dst:
        dst.writerecords(chunk)
    # Create a flag file to indicate this chunk is finished
    with open(output_path.replace(".geojson", ".finish"), "w") as f:
        f.write("Processing complete.\n")

CHUNKSIZE = 10000000
def chunksize_vector(input_vector, chunk_temp_folder):
    """Splits a large GeoJSON file into smaller chunks and prevents duplicate processing."""
    chunk_files = []
    chunk = []
    chunk_ind = 0

    os.makedirs(chunk_temp_folder, exist_ok=True)  # Ensure the output folder exists

    chunk_flag = os.path.join(chunk_temp_folder, "finished.txt")
    if os.path.exists(chunk_flag):
        return glob(os.path.join(chunk_temp_folder, "*.geojson"))

    with fiona.open(input_vector, "r") as f:
        schema, crs = f.schema, f.crs  # Extract schema and CRS once

        for feature in f:
            chunk.append(feature)
            if len(chunk) >= CHUNKSIZE:
                chunk_ind += 1
                chunk_path = os.path.join(chunk_temp_folder, f"{chunk_ind:02d}.geojson")
                flag_path = chunk_path.replace(".geojson", ".finish")

                if os.path.exists(flag_path):
                    logging.info(f"{input_vector}: Skipping {chunk_path}, already processed.")
                else:
                    write_chunk(chunk, chunk_path, schema, crs)
                chunk_files.append(chunk_path)

                chunk.clear()  # Efficiently reset chunk

        # Write the last chunk if any data remains
        if chunk:
            chunk_ind += 1
            chunk_path = os.path.join(chunk_temp_folder, f"{chunk_ind:02d}.geojson")
            flag_path = chunk_path.replace(".geojson", ".finish")

            if os.path.exists(flag_path):
                logging.info(f"{input_vector}: Skipping {chunk_path}, already processed.")
            else:
                write_chunk(chunk, chunk_path, schema, crs)
            chunk_files.append(chunk_path)

    with open(chunk_flag, "w") as f:
        f.writelines("finished.\n")
    logging.info(f"{input_vector}: Processed {chunk_ind} chunks (new + existing).")
    return chunk_files
        
def process_single_geojson(input_vector, continent):
    try:
        height_result_dir = f"/dss/lxclscratch/03/ga27lib2/global-building-height-results/{continent.lower()}"
        logging.info(f"=== Processing {input_vector} ===")
        input_basename = os.path.basename(input_vector)
        input_folder = glob(os.path.join(height_result_dir, input_basename[:31] + "*"))


        output_folder = os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/lod1_max2/", continent) #MAX
        os.makedirs(output_folder, exist_ok=True)
        output_temp_folder = os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/lod1_temp_max2/", continent, input_basename.split(".")[0]) #MAX
        os.makedirs(output_temp_folder, exist_ok=True)
        bf_temp_folder = os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/bf_temp_max2/", continent, input_basename.split(".")[0]) #MAX
        os.makedirs(bf_temp_folder, exist_ok=True)
        chunk_temp_folder = os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/input_temp2/", continent, input_basename.split(".")[0])
        os.makedirs(chunk_temp_folder, exist_ok=True)
        height_result_temp_dir = os.path.join("/dss/lxclscratch/03/ga27lib2/fahong_output/gbhr_temp2/", continent, input_basename.split(".")[0])
        os.makedirs(height_result_temp_dir, exist_ok=True)
        tic = time.time()

        output_file = os.path.join(output_folder, input_basename.split(".")[0]+"_lod1.geojson")
        output_flag = output_file.replace(".geojson", ".finish")
        output_fail = output_file.replace(".geojson", ".fail")

        if os.path.exists(output_flag):
            return

        if not input_folder:
            print(f"{input_vector}: Input folder not found for {input_basename}")
            return

        input_folder = input_folder[0]
        target_crs = CRS.from_epsg(3857)

        match = re.search(r'([ew])(\d{3})_([ns])(\d{2})_([ew])(\d{3})_([ns])(\d{2})', input_vector)
        if not match:
            return  # Skip if not a match

        # Parse longitudes and latitudes
        lonmin = float(match.group(2)) if match.group(1) == 'e' else -float(match.group(2))
        lonmax = float(match.group(6)) if match.group(5) == 'e' else -float(match.group(6))
        latmax = float(match.group(4)) if match.group(3) == 'n' else -float(match.group(4))
        latmin = float(match.group(8)) if match.group(7) == 'n' else -float(match.group(8))

        bbox_shapes = []
        bbox_names = []

        bbox_names_total = []
        bbox_extents_total = []
        for x_ind in range(25):
            for y_ind in range(25):
                xmin = lonmin + x_ind * 0.2
                ymin = latmin + y_ind * 0.2

                xmax = xmin + 0.2
                ymax = ymin + 0.2

                com_file = os.path.join(output_temp_folder, f"{xmin:.1f}_{ymax:.1f}_{xmax:.1f}_{ymin:.1f}_sr_nu.com")
                finish_file = os.path.join(output_temp_folder, f"{xmin:.1f}_{ymax:.1f}_{xmax:.1f}_{ymin:.1f}_sr_ss.finish")

                if os.path.exists(com_file):
                    continue
                if os.path.exists(finish_file):
                    continue

                bbox_name = os.path.join(bf_temp_folder, f"{xmin:.1f}_{ymax:.1f}_{xmax:.1f}_{ymin:.1f}_sr_ss.geojson")
                bbox_flag = bbox_name.replace(".geojson", ".finish")
                if os.path.exists(bbox_flag):
                    bbox_names_total.append(bbox_name)
                    bbox_extents_total.append((xmin, ymin, xmax, ymax))
                else:
                    xxmin, yymin = transformer.transform(xmin, ymin)
                    xxmax, yymax = transformer.transform(xmax, ymax)
                    bbox = box(xxmin, yymin, xxmax, yymax)

                    bbox_shapes.append(bbox)
                    bbox_names.append(bbox_name)
                    bbox_names_total.append(bbox_name)
                    bbox_extents_total.append((xmin, ymin, xmax, ymax))


        bbox_bounds = np.array([b.bounds for b in bbox_shapes])
        logging.info(f"{input_vector}: processing {len(bbox_names_total)} bboxes!")

        input_chunk_vectors = chunksize_vector(input_vector, chunk_temp_folder)
        logging.info(f"{input_vector}: Input file divided into {len(input_chunk_vectors)} chunks!")

        for chunk_ind, input_chunk_vector in enumerate(input_chunk_vectors):
            chunk_processing_flag = os.path.join(output_temp_folder, f"{chunk_ind:02d}.finish")
            if os.path.exists(chunk_processing_flag):
                continue

            logging.info(f"{input_vector}: Processing chunk {chunk_ind:02d}...")
            feat_sources, feat_shapes, feat_bounds, feat_ids, feat_regions = get_source_shapes_and_bounds(input_chunk_vector)
            coarse_intersection_flag = get_coarse_intersection_flag(bbox_bounds, feat_bounds)

            logging.info(f"{input_vector}: Intersection flag array {coarse_intersection_flag.shape} got!")

            for i, bbox_name in enumerate(bbox_names):
                bbox_name = bbox_name.replace("_sr_ss.geojson", f"_{chunk_ind:02d}_sr_ss.geojson")
                if os.path.exists(bbox_name.replace(".geojson", ".finish")):
                    continue
                write_cropped_features(feat_sources, feat_shapes, feat_ids, feat_regions, bbox_shapes[i], np.nonzero(coarse_intersection_flag[i])[0], bbox_name)
            
            for i, bbox_name in enumerate(bbox_names_total):
                bbox_name = bbox_name.replace("_sr_ss.geojson", f"_{chunk_ind:02d}_sr_ss.geojson")
                xmin, ymin, xmax, ymax = bbox_extents_total[i]
                raster_chunk_file_ss = os.path.join(input_folder, f"{xmin:.1f}_{ymax:.1f}_{xmax:.1f}_{ymin:.1f}_sr_ss.tif")
                if not os.path.exists(raster_chunk_file_ss):
                    bbox_base_name = os.path.basename(bbox_name).replace("_ss.geojson", "_nu.geojson")
                    bbox_out_folder = os.path.join(output_temp_folder, bbox_base_name.split(".")[0])
                    os.makedirs(bbox_out_folder, exist_ok=True)
                    bbox_out_path = os.path.join(bbox_out_folder, f"{chunk_ind:02d}.geojson")
                    bbox_out_flag = bbox_out_path.replace(".geojson", ".com")

                    if os.path.exists(bbox_out_flag):
                        continue

                    with fiona.open(bbox_name, "r") as src_vector, \
                        fiona.open(bbox_out_path, "w", schema=SCHEMA, crs="EPSG:3857") as dst_vector:

                        for feature in src_vector:
                            feature["properties"].update({
                                "height": -999,
                                "var": -999
                            })
                            dst_vector.write(feature)
                    with open(bbox_out_flag, "w") as f:
                        f.writelines("Finished!\n")
                else:
                    raster_chunk_file_var = raster_chunk_file_ss.replace("_sr_ss.tif", "_sr_var.tif")
                    bbox_out_folder = os.path.join(output_temp_folder, os.path.basename(raster_chunk_file_ss).split(".tif")[0])
                    os.makedirs(bbox_out_folder, exist_ok=True)
                    output_vector = os.path.join(bbox_out_folder, f"{chunk_ind:02d}.geojson")
                    output_vector_flag = output_vector.replace(".geojson", ".finish")

                    if os.path.exists(output_vector_flag):
                        continue

                    temp_raster_chunk_ss_file = os.path.join(height_result_temp_dir, os.path.basename(raster_chunk_file_ss))
                    temp_raster_chunk_var_file = os.path.join(height_result_temp_dir, os.path.basename(raster_chunk_file_var))

                    if (not os.path.exists(temp_raster_chunk_ss_file)) | (not os.path.exists(temp_raster_chunk_var_file)):
                        with rasterio.open(raster_chunk_file_ss) as src_ss_raster, \
                            rasterio.open(raster_chunk_file_var) as src_var_raster:

                            transform, width, height = calculate_default_transform(
                                src_ss_raster.crs, target_crs, src_ss_raster.width, src_ss_raster.height, *src_ss_raster.bounds
                            )

                            kwargs = src_ss_raster.meta.copy()
                            kwargs.update({
                                "crs": target_crs,
                                "transform": transform,
                                "width": width,
                                "height": height
                            })

                            with rasterio.open(temp_raster_chunk_ss_file, "w", **kwargs) as dst:
                                for i in range(1, src_ss_raster.count + 1):
                                    reproject(
                                        source=rasterio.band(src_ss_raster, i),
                                        destination=rasterio.band(dst, i),
                                        src_transform=src_ss_raster.transform,
                                        src_crs=src_ss_raster.crs,
                                        dst_transform=transform,
                                        dst_crs=target_crs,
                                        resampling=Resampling.nearest
                                    )

                            with rasterio.open(temp_raster_chunk_var_file, "w", **kwargs) as dst:
                                for i in range(1, src_var_raster.count + 1):
                                    reproject(
                                        source=rasterio.band(src_var_raster, i),
                                        destination=rasterio.band(dst, i),
                                        src_transform=src_var_raster.transform,
                                        src_crs=src_var_raster.crs,
                                        dst_transform=transform,
                                        dst_crs=target_crs,
                                        resampling=Resampling.nearest
                                    )

                    with fiona.open(bbox_name, "r") as src_vector, \
                        rasterio.open(temp_raster_chunk_ss_file) as src_ss_raster, \
                        rasterio.open(temp_raster_chunk_var_file) as src_var_raster, \
                        fiona.open(output_vector, "w", schema=SCHEMA, crs="EPSG:3857") as dst_vector:
                        
                        for feature in src_vector:
                            masked_data, _ = mk.mask(src_ss_raster, [feature["geometry"]], crop=True, all_touched=True, nodata=-1)
                            masked_var, _ = mk.mask(src_var_raster, [feature["geometry"]], crop=True, all_touched=True, nodata=-1)

                            median_value = np.max(masked_data) #MAX
                            closest_index = np.argmin(np.abs(masked_var.flatten() - median_value))
                            var_value = masked_var.flatten()[closest_index]

                            feature["properties"].update({
                                "height": float(median_value),
                                "var": float(var_value)
                            })

                            dst_vector.write(feature)
                    # Cleanup: Remove temporary files
                    os.remove(temp_raster_chunk_ss_file)
                    os.remove(temp_raster_chunk_var_file)
                    
                    with open(output_vector_flag, "w") as f:
                        f.writelines("Finished!\n")

            with open(chunk_processing_flag, "w") as f:
                f.writelines("finished!\n")

        logging.info(f"{input_vector}: Merging chunked results of {len(bbox_names_total)} bboxes...")
        for i, bbox_name in enumerate(bbox_names_total):
            xmin, ymin, xmax, ymax = bbox_extents_total[i]
            raster_chunk_folder_ss = os.path.join(output_temp_folder, f"{xmin:.1f}_{ymax:.1f}_{xmax:.1f}_{ymin:.1f}_sr_ss")
            raster_chunk_output_file = raster_chunk_folder_ss+".geojson"
            raster_chunk_output_flag = raster_chunk_folder_ss+".finish"

            if not os.path.exists(raster_chunk_folder_ss):
                raster_chunk_folder_ss = os.path.join(output_temp_folder, f"{xmin:.1f}_{ymax:.1f}_{xmax:.1f}_{ymin:.1f}_sr_nu")
                raster_chunk_output_file = raster_chunk_folder_ss+".geojson"
                raster_chunk_output_flag = raster_chunk_folder_ss+".com"

            with fiona.open(raster_chunk_output_file, "w", crs="EPSG:3857", schema=SCHEMA) as dst_chunk:
                for temp_file in glob(os.path.join(raster_chunk_folder_ss, "*.geojson")):
                    with fiona.open(temp_file, "r") as src_chunk:
                        for feature in src_chunk:
                            dst_chunk.write(feature)

            with open(raster_chunk_output_flag, "w") as temp_f:
                temp_f.writelines("\n")

        logging.info(f"{input_vector}: Merging final results ...")
        merge_single_result(input_vector, continent)

        toc = time.time()
        logging.info(f"{input_vector}: Finished processing {input_vector} in {toc - tic:.2f} seconds.")
    except Exception as e:
        with open(output_fail, "w") as f:
            f.writelines("Failed! \n")
        logging.error(f"{input_vector}: Error processing {input_vector} at line {traceback.extract_tb(e.__traceback__)[0].lineno}: {e}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoJSON files into LoD-1, input file specified version.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the merged GeoJSON folders")
    parser.add_argument("--processes", default=0, type=int, help="Number of processes")
    args = parser.parse_args()

    with open(input_file, "r") as f:
        lines = f.readlines()

    input_vectors = [line.rstrip() for line in lines]
    continents = [tfl.split("/")[7] for tfl in input_vectors]
    
    if args.processes > 0:
        processes = args.processes
    else:
        processes = min(multiprocessing.cpu_count(), len(input_vectors))

    with multiprocessing.Pool(processes=processes) as pool:
        logging.info(f"Start processing {len(input_vectors)} vectors with {processes} processes.")
        pool.starmap(process_single_geojson, [(input_vector, continent) for input_vector, continent in zip(input_vectors, continents)])
