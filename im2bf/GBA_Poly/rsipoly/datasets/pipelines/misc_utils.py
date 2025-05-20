from pathlib import Path
from osgeo import gdal
import numpy as np
import json
import os
from tqdm import tqdm
from PIL import Image


UNITS_PER_METER_CONVERSION_FACTORS = {"cm": 100.0, "m": 1.0}


def save_image(img_path, img):
    """Note this function does not utilize conversion factors to meters, so implicitly assumes units are in meters."""
    rows = img.shape[0]
    cols = img.shape[1]
    bands = 1
    if np.ndim(img) == 3:
        bands = img.shape[2]
    if img.dtype == np.uint8:
        data_type = gdal.GDT_Byte
    else:
        data_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff").Create(
        str(img_path), rows, cols, bands, data_type, ["COMPRESS=LZW"]
    )
    if bands == 1:
        driver.GetRasterBand(1).WriteArray(img[:, :])
    else:
        for i in range(bands):
            driver.GetRasterBand(i + 1).WriteArray(img[:, :, i])
    driver.FlushCache()
    driver = None


def load_image(
    image_path,
    args,
    dtype_out="float32",
    units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
):

    image_path = Path(image_path)
    if not image_path.exists():
        return None
    image = gdal.Open(str(image_path))
    image = image.ReadAsArray()

    # convert AGL units and fill nan placeholder with nan
    if "AGL" in image_path.name:
        image = image.astype(dtype_out)
        np.putmask(image, image == args.nan_placeholder, np.nan)
        # e.g., (cm) / (cm / m) = m
        units_per_meter = units_per_meter_conversion_factors[args.unit]
        image = (image / units_per_meter).astype(dtype_out)

    # transpose if RGB
    if len(image.shape) == 3:
        image = np.transpose(image, [1, 2, 0])

    return image


def load_vflow(
    vflow_path,
    agl,
    args,
    dtype_out="float32",
    units_per_meter_conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
    return_vflow_pred_mat=False,
):

    vflow_path = Path(vflow_path)
    vflow_data = json.load(vflow_path.open("r"))

    # e.g., (pixels / cm) * (cm / m) = (pixels / m)
    units_per_meter = units_per_meter_conversion_factors[args.unit]
    vflow_data["scale"] = vflow_data["scale"] * units_per_meter

    xdir, ydir = np.sin(vflow_data["angle"]), np.cos(vflow_data["angle"])
    mag = agl * vflow_data["scale"]
    
    vflow_items = [mag.astype(dtype_out), xdir.astype(dtype_out), ydir.astype(dtype_out), vflow_data]

    if return_vflow_pred_mat:
        vflow = np.zeros((agl.shape[0],agl.shape[1],2))
        vflow[:,:,0] = mag * xdir
        vflow[:,:,1] = mag * ydir
        vflow_items.insert(0, vflow)

    return vflow_items


def get_r2(error_sum, gt_sq_sum, data_sum, count):
    return 1 - error_sum / (gt_sq_sum - (data_sum ** 2) / count)


def get_rms(errors):
    return np.sqrt(np.mean(np.square(errors)))


def get_angle_error(dir_pred, dir_gt):
    dir_pred /= np.linalg.norm(dir_pred)
    dir_gt /= np.linalg.norm(dir_gt)
    cos_ang = np.dot(dir_pred, dir_gt)
    sin_ang = np.linalg.norm(np.cross(dir_pred, dir_gt))
    rad_diff = np.arctan2(sin_ang, cos_ang)
    angle_error = np.degrees(rad_diff)
    return angle_error


def get_r2_info(data_gt, data_pred):

    data_pred = np.squeeze(data_pred)

    not_nan = ~np.isnan(data_gt)
    count = np.sum(not_nan)
    diff = data_pred - data_gt

    error_sum = np.sum(np.square(diff[not_nan]))
    rms = get_rms(diff[not_nan])
    data_sum = np.sum(data_gt[not_nan])
    gt_sq_sum = np.sum(np.square(data_gt[not_nan]))

    return count, error_sum, rms, data_sum, gt_sq_sum


def convert_and_compress_prediction_dir(
    predictions_dir,
    to_unit="cm",
    agl_dtype="uint16",
    compression_type="tiff_adobe_deflate",
    conversion_factors=UNITS_PER_METER_CONVERSION_FACTORS,
):
    """Convert and compress prediction directory.
    Default parameters are consistent with DrivenData platform submission requirements.
    """
    predictions_dir = Path(predictions_dir)
    converted_predictions_dir = predictions_dir.with_name(
        f"{predictions_dir.name}_converted_{to_unit}_compressed_{agl_dtype}"
    )
    converted_predictions_dir.mkdir(exist_ok=True, parents=True)

    conversion_factor = conversion_factors[to_unit]

    agl_paths = list(predictions_dir.glob("*_AGL.tif"))
    json_paths = list(
        pth.with_name(pth.name.replace("_AGL", "_VFLOW")).with_suffix(".json")
        for pth in agl_paths
    )
    for agl_path, json_path in tqdm(zip(agl_paths, json_paths), total=len(agl_paths)):
        # convert and compress agl tif
        imarray = np.array(Image.open(agl_path))
        imarray = np.round(imarray * conversion_factor).astype(agl_dtype)
        new_image = Image.fromarray(imarray)
        new_image_path = converted_predictions_dir / agl_path.name
        new_image.save(str(new_image_path), "TIFF", compression=compression_type)

        # convert and compress vflow json
        vflow = json.load(json_path.open("r"))
        vflow["scale"] = vflow["scale"] / conversion_factor
        new_json_path = converted_predictions_dir / json_path.name
        json.dump(vflow, new_json_path.open("w"))


