from glob import glob
import os
import re
import fiona
import numpy as np
import time
from tqdm import tqdm
import logging
import traceback
from shapely.geometry import mapping
import multiprocessing
from shapely.geometry import box
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
    no_crop_copy
)
import argparse
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'
def switch_log_file(logger, new_file):
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    new_handler = logging.FileHandler(new_file)
    new_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(new_handler)

def process_geojson(geojson_ours, district_folder, cache_dir, output_dir, main_source="osm"):
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
        
        if not os.path.exists(geojson_ours):
            geojson_ours = None

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

                if os.path.exists(crop_flag):
                    continue

                if roi_ids_todo:
                    roi_bounds_todo = np.array(roi_bounds_todo)
                    logging.info(f"{geojson_ours_name}: {name_source} Processing {len(roi_ids_todo)} RoIs...")
                
                    feat_shapes, feat_bounds, feat_ids = get_source_shapes_and_bounds(geojson_source, name_source)
                    logging.info(f"{geojson_ours_name}: {name_source}: Processing {len(feat_ids)} features...")
                    if len(feat_ids) > 0:
                        coarse_intersection_flag = get_coarse_intersection_flag(roi_bounds_todo, feat_bounds)
                        logging.info(f"{geojson_ours_name}: {name_source}: Coarse intersection flag matrix {coarse_intersection_flag.shape} got ...")

                        for roi_index in range(len(roi_shapes_todo)):
                            roi_geom = roi_shapes_todo[roi_index]
                            roi_id = roi_ids_todo[roi_index]
                            output_folder = os.path.join(cropped_file_dir, roi_id)
                            output_flag = os.path.join(output_folder, name_source + ".finish")
                            if os.path.exists(output_flag):
                                continue

                            os.makedirs(output_folder, exist_ok=True)
                            logging.info(f"{geojson_ours_name}: {name_source}: RoI {roi_id}: Processing {coarse_intersection_flag[roi_index].sum()} features...")
                            
                            write_cropped_features(feat_shapes, feat_ids, roi_geom, np.nonzero(coarse_intersection_flag[roi_index])[0], name_source, os.path.join(output_folder, name_source + ".geojson"))
                            
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
        logging.info(f"{geojson_ours_name}: Cropping finished!")
        logging.info(f"{geojson_ours_name}: Aggregating different sources ...")
        
        # Merging results and aggregating files
        cropped_file_dirs = []
        if geojson_district:
            for roi_id in roi_ids_total:
                cropped_file_dirs.append(os.path.join(cropped_file_dir, roi_id))
        else:
            cropped_file_dirs = [os.path.join(cropped_file_dir, "total")]

        if (main_source == "google") & (geojson_google is None):
            main_source = "osm"

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
    output_dirs = ["/dss/lxclscratch/03/ga27lib2/ga27lib2/outputs3/"+line.split("/")[6] for line in lines]
    if args.processes == 0:
        num_workers = min(len(geojson_ours), multiprocessing.cpu_count())
    else:
        num_workers = args.processes
    logging.info(f"Processing with {num_workers} cpus!")

    with multiprocessing.Pool(processes=num_workers) as pool:  # Adjust based on available resources
        pool.starmap(process_geojson, [(geojson_our, district_folder, cache_dir, output_dir, args.main_source) for geojson_our, output_dir in zip(geojson_ours, output_dirs)])
