from glob import glob
import os
import re
import fiona
import numpy as np
import time
from tqdm import tqdm
import logging
import traceback
from shapely.geometry import mapping, box
import multiprocessing
from utils import (
    get_bounds_in_4326,
    query_osm_bf,
    query_ms_bf,
    query_google_bf,
    query_3dglobfp_bf,
    get_roi_shapes_and_bounds,
    get_source_shapes_and_bounds,
    get_coarse_intersection_flag,
    write_cropped_features,
    get_best_without_osm,
    get_best_with_osm,
    aggregate_osm_and_source,
    merge_all_results,
    no_crop_copy,
    chunked_processing_cropping,
    clear_all_chunk_results
)
from concurrent.futures import ThreadPoolExecutor

import argparse
CHUNKSIZE=1000000
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'
def switch_log_file(logger, new_file):
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    new_handler = logging.FileHandler(new_file)
    new_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(new_handler)

def process_geojson(geojson_ours, district_folder, cache_dir, output_dir, main_source):
    try:    
        logging.info(f"Processing {geojson_ours}")
        geojson_ours_name = os.path.basename(geojson_ours)
        switch_log_file(logging.getLogger(), f"srun_outputs/{geojson_ours_name[:31]}.log")
        logging.info("=============================================================================")

        fail_flag = os.path.join(output_dir, geojson_ours_name.split(".")[0] + ".fail")
        flag_file = os.path.join(output_dir, geojson_ours_name.split(".")[0] + ".finish")

        if os.path.exists(fail_flag) | os.path.exists(flag_file):
            logging.error(f"{geojson_ours_name} exists! Skipping...")
            return

        match = re.search(r'([ew])(\d{3})_([ns])(\d{2})_([ew])(\d{3})_([ns])(\d{2})', geojson_ours_name)
        if not match:
            return  # Skip if not a match
        
        # Parse longitudes and latitudes
        lonmin = float(match.group(2)) if match.group(1) == 'e' else -float(match.group(2))
        lonmax = float(match.group(6)) if match.group(5) == 'e' else -float(match.group(6))
        latmax = float(match.group(4)) if match.group(3) == 'n' else -float(match.group(4))
        latmin = float(match.group(8)) if match.group(7) == 'n' else -float(match.group(8))
        bbox = (latmax, latmin, lonmax, lonmin)
        bbox_shape = box(lonmin, latmin, lonmax, latmax)
        tic = time.time()

        geojson_osm = query_osm_bf(bbox)
        geojson_ms = query_ms_bf(bbox)
        geojson_google = query_google_bf(bbox)
        geojson_3dglobfp = query_3dglobfp_bf(geojson_ours_name[:31])

        cropped_file_dir = os.path.join(cache_dir, geojson_ours_name[:31])
        os.makedirs(cropped_file_dir, exist_ok=True)

        logging.info(f"{geojson_ours_name}: Query finished!")
        geojson_district = glob(os.path.join(district_folder, geojson_ours_name[:31] + "*.geojson"))
        
        logging.info(f"{geojson_ours_name}: Cropping into districts...")
        if geojson_district:
            geojson_district = geojson_district[0]
            roi_shapes, roi_bounds, roi_ids = get_roi_shapes_and_bounds(geojson_district, bbox_shape)

            for geojson_source, name_source in zip(
                [geojson_ours, geojson_ms, geojson_google, geojson_osm, geojson_3dglobfp],
                ["ours2", "ms", "google", "osm", "3dglobfp"]
            ):
                logging.info(f"{geojson_ours_name}: Cropping {name_source} into districts...")
                if geojson_source is None:
                    continue
                crop_flag = os.path.join(cropped_file_dir, name_source + ".finish")

                roi_shapes_todo = []
                roi_bounds_todo = []
                roi_ids_todo = []
                roi_ids_total = []
                existing_rois = [f for f in os.listdir(cropped_file_dir) if os.path.isdir(os.path.join(cropped_file_dir, f))]
                for roi_shape, roi_bound, roi_id in zip(roi_shapes, roi_bounds, roi_ids):
                    output_flag = os.path.join(cropped_file_dir, roi_id, name_source+".finish")
                    roi_ids_total.append(roi_id)
                    if not os.path.exists(output_flag):
                        roi_shapes_todo.append(roi_shape)
                        roi_bounds_todo.append(roi_bound)
                        roi_ids_todo.append(roi_id)
                to_delete_roi_dirs = [os.path.join(cropped_file_dir, f) for f in existing_rois if f not in roi_ids_total]
                logging.info(f"{geojson_ours_name}: {len(to_delete_roi_dirs)} should be deleted!")

                for to_delete_roi_dir in to_delete_roi_dirs:
                    shutil.rmtree(to_delete_roi_dir)
                output_folder = cropped_file_dir
                if os.path.exists(crop_flag):
                    continue
                if roi_ids_todo:
                    roi_bounds_todo = np.array(roi_bounds_todo)
                    logging.info(f"{geojson_ours_name}: {name_source} Processing {len(roi_ids_todo)} RoIs...")
                    with fiona.open(geojson_source, "r") as f:
                        num_chunks = len(f) // CHUNKSIZE + 1
                        logging.info(f"{geojson_source}: {name_source}: Processing in {num_chunks} chunks...")

                        chunk = []
                        chunk_ind = 0

                        # Process features in chunks
                        for feature in f:
                            chunk.append(feature)
                            if len(chunk) >= CHUNKSIZE:
                                chunk_ind += 1
                                chunk_output_flag = os.path.join(output_folder, f"chunk_{name_source}_{chunk_ind:05d}.finish")
                                
                                # Check if the chunk has already been processed
                                if not os.path.exists(chunk_output_flag):
                                    logging.info(f"{geojson_ours_name}: {name_source}: Processing chunk {chunk_ind}/{num_chunks}...")
                                    chunked_processing_cropping(chunk, chunk_ind, roi_ids_todo, roi_shapes_todo, roi_bounds_todo, output_folder, name_source, f.crs)
                                else:
                                    logging.info(f"{geojson_ours_name}: {name_source}: Skipping chunk {chunk_ind}/{num_chunks}, already processed.")

                                chunk = []  # Reset chunk

                        # Handle the last chunk
                        if chunk:
                            chunk_ind += 1
                            chunk_output_flag = os.path.join(output_folder, f"chunk_{name_source}_{chunk_ind:05d}.finish")
                            if not os.path.exists(chunk_output_flag):
                                logging.info(f"{geojson_ours_name}: {name_source}: Processing final chunk {chunk_ind}/{num_chunks}...")
                                chunked_processing_cropping(chunk, chunk_ind, roi_ids_todo, roi_shapes_todo, roi_bounds_todo, output_folder, name_source, f.crs)
                            else:
                                logging.info(f"{geojson_ours_name}: {name_source}: Skipping final chunk {chunk_ind}/{num_chunks}, already processed.")

                    logging.info(f"{geojson_source}: {name_source}: Processing completed.")

                    for roi_id in roi_ids_todo:
                        output_path = os.path.join(cropped_file_dir, roi_id, name_source+".geojson")
                        input_files = glob(os.path.join(output_folder, roi_id, "chunk*", "chunk_"+name_source+".geojson"))
                        merge_all_results(input_files, output_path)

                        with open(output_path.replace(".geojson", ".finish"), "w") as f:
                            f.writelines("finished!\n")

                with open(crop_flag, "w") as f:
                    f.write("finished!\n")
        else:
            for geojson_source, name_source in zip(
                [geojson_ours, geojson_ms, geojson_google, geojson_osm, geojson_3dglobfp],
                ["ours2", "ms", "google", "osm", "3dglobfp"]
            ):
                if geojson_source is None:
                    continue
                crop_flag = os.path.join(cropped_file_dir, name_source + ".finish")
                if os.path.exists(crop_flag):
                    continue

                output_folder = os.path.join(cropped_file_dir, "total")
                os.makedirs(output_folder, exist_ok=True)
                cropped_output_file = os.path.join(output_folder, name_source + ".geojson")

                no_crop_copy(geojson_source, cropped_output_file, name_source)

                with open(crop_flag, "w") as f:
                    f.write("finished!\n")     
        if not os.path.exists(crop_flag):    
            clear_all_chunk_results(output_folder)
        logging.info(f"{geojson_ours_name}: Cropping finished!")
        logging.info(f"{geojson_ours_name}: Aggregating different sources ...")
        # Merging results and aggregating files

        cropped_file_dirs = []
        if geojson_district:
            for roi_id in roi_ids_total:
                cropped_file_dirs.append(os.path.join(cropped_file_dir, roi_id))
        else:
            cropped_file_dirs = [os.path.join(cropped_file_dir, "total")]
        for file_dir in cropped_file_dirs:
            if not os.path.isdir(file_dir):
                continue
            finish_file = os.path.join(file_dir, "finished.txt")
            if os.path.exists(finish_file):
                continue

            osm_file = os.path.join(file_dir, main_source+".geojson") if os.path.exists(os.path.join(file_dir, main_source+".geojson")) else None
            cropped_merged_file = os.path.join(file_dir, "merged.geojson")
            if osm_file is None:
                best_source, best_source_name, best_diff = get_best_without_osm(file_dir, main_source)
            else:
                best_source, best_source_name, best_diff = get_best_with_osm(file_dir, main_source)

            aggregate_osm_and_source(osm_file, best_source, best_diff, best_source_name, cropped_merged_file, main_source)
        logging.info(f"{geojson_ours_name}: Aggregating finished!")
        logging.info(f"{geojson_ours_name}: Merging ...")

        result_list = [os.path.join(cropped_file_dir, "merged.geojson") for cropped_file_dir in cropped_file_dirs]
        output_file = os.path.join(output_dir, geojson_ours_name.split(".")[0] + ".geojson")
        if not os.path.exists(flag_file):
            merge_all_results(result_list, output_file)
            with open(flag_file, "w") as f:
                f.writelines("Finished! \n")
        logging.info(f"{geojson_ours_name}: Merging finished!")

        toc = time.time()
        logging.info(f"Finished processing {geojson_ours_name} in {toc - tic:.2f} seconds.")
    except Exception as e:
        with open(fail_flag, "w") as f:
            f.writelines("Failed! \n")
        logging.error(f"Error processing {geojson_ours_name} at line {traceback.extract_tb(e.__traceback__)[0].lineno}: {e}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoJSON files in parallel, File List Specifed Version")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file list text file")
    parser.add_argument("--main_source", default="osm", type=str, help="Main source for processing")
    parser.add_argument("--processes", default=0, type=int, help="Number of processes")
    args = parser.parse_args()

    district_folder = "/dss/lxclscratch/03/ga27lib2/boundary_tiles"
    cache_dir = "/dss/lxclscratch/03/ga27lib2/ga27lib2/caches"
    
    with open(args.input_file, "r") as f:
        lines = f.readlines()
    
    geojson_ours = [line.rstrip() for line in lines]
    output_dirs = ["/dss/lxclscratch/03/ga27lib2/ga27lib2/outputs2/"+line.split("/")[6] for line in lines]
    if args.processes == 0:
        num_workers = min(multiprocessing.cpu_count(), len(geojson_ours))
    else:
        num_workers = args.processes
    logging.info(f"Processing with {num_workers} cpus!")

    with multiprocessing.Pool(processes=num_workers) as pool:  # Adjust based on available resources
        pool.starmap(process_geojson, [(geojson_our, district_folder, cache_dir, output_dir, args.main_source) for geojson_our, output_dir in zip(geojson_ours, output_dirs)])