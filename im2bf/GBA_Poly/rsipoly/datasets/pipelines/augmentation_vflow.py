import os
import math
import numpy as np
import cv2

from .invert_flow import invert_flow


RNG = np.random.RandomState(1234)


def augment_vflow(
    image,
    mag,
    xdir,
    ydir,
    angle,
    scale,
    agl=None,
    rotate_prob=0.3,
    flip_prob=0.3,
    scale_prob=0.3,
    agl_prob=0.3,
    rng=RNG,
):
    # increase heights
    if np.isnan(mag).any() or np.isnan(agl).any():
        agl_prob = 0
    if rng.uniform(0, 1) < agl_prob:
        max_agl = np.nanmax(agl)
        max_building_agl = 200.0
        max_factor = 2.0
        max_scale_agl = min(max_factor, (max_building_agl / max_agl))
        scale_height = rng.uniform(1.0, max(1.0, max_scale_agl))
        image, mag, agl = warp_agl(image, mag, angle, agl, scale_height, max_factor)
    # rotate
    if rng.uniform(0, 1) < rotate_prob:
        rotate_angle = rng.randint(0, 359)
        xdir, ydir = rotate_xydir(xdir, ydir, rotate_angle)
        image, mag, agl = rotate_image(image, mag, agl, rotate_angle)
    # x flip
    if rng.uniform(0, 1) < flip_prob:
        image, mag, agl = flip(image, mag, agl, dim="x")
        xdir *= -1
    # y flip
    if rng.uniform(0, 1) < flip_prob:
        image, mag, agl = flip(image, mag, agl, dim="y")
        ydir *= -1
    # rescale
    if rng.uniform(0, 1) < scale_prob:
        factor = 1.0 - 0.3 * rng.random()
        image, mag, agl, scale = rescale_vflow(image, mag, agl, scale, factor)
    return image, mag, xdir, ydir, agl, scale


def flip(image, mag, agl, dim):
    if dim == "x":
        image = image[:, ::-1, :]
        mag = mag[:, ::-1]
        if agl is not None:
            agl = agl[:, ::-1]
    elif dim == "y":
        image = image[::-1, :, :]
        mag = mag[::-1, :]
        if agl is not None:
            agl = agl[::-1, :]
    return image, mag, agl


def get_crop_region(image_rotated, image):
    excess_buffer = np.array(image_rotated.shape[:2]) - np.array(image.shape[:2])
    r1, c1 = (excess_buffer / 2).astype(np.int)
    r2, c2 = np.array([r1, c1]) + image.shape[:2]
    return r1, c1, r2, c2


def rotate_xydir(xdir, ydir, rotate_angle):
    base_angle = np.degrees(np.arctan2(xdir, ydir))
    xdir = np.sin(np.radians(base_angle + rotate_angle))
    ydir = np.cos(np.radians(base_angle + rotate_angle))
    return xdir, ydir


def rotate_image(image, mag, agl, angle, image_only=False):
    if image_only:
        h, w = image.shape[:2]
    else:
        h, w = mag.shape[:2]
    rw, rh = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D((rw, rh), angle, 1.0)
    cos, sin = np.abs(rot_mat[0, 0:2])
    wnew = int((h * sin) + (w * cos))
    hnew = int((h * cos) + (w * sin))
    rot_mat[0, 2] += np.int((wnew / 2) - rw)
    rot_mat[1, 2] += np.int((hnew / 2) - rh)
    image_rotated = (
        None
        if image is None
        else cv2.warpAffine(image, rot_mat, (wnew, hnew), flags=cv2.INTER_LINEAR)
    )
    if image_rotated is not None:
        r1, c1, r2, c2 = get_crop_region(image_rotated, image)
        image_rotated = image_rotated[r1:r2, c1:c2, :]
    if image_only:
        return image_rotated
    agl_rotated = (
        None
        if agl is None
        else cv2.warpAffine(agl, rot_mat, (wnew, hnew), flags=cv2.INTER_NEAREST)
    )
    mag_rotated = cv2.warpAffine(mag, rot_mat, (wnew, hnew), flags=cv2.INTER_NEAREST)
    if image_rotated is None:
        r1, c1, r2, c2 = get_crop_region(mag_rotated, mag)
    mag_rotated = mag_rotated[r1:r2, c1:c2]
    if agl_rotated is not None:
        agl_rotated = agl_rotated[r1:r2, c1:c2]
    return image_rotated, mag_rotated, agl_rotated


def rescale(image, factor, fill_value=0):
    output_shape = np.copy(image.shape)
    target_shape = (int(image.shape[0] * factor), int(image.shape[1] * factor))
    image = cv2.resize(image, target_shape, interpolation=cv2.INTER_NEAREST)
    image = np.expand_dims(image, axis=2)
    if factor > 1.0:
        start = int((target_shape[0] - output_shape[0]) / 2.0)
        end = start + output_shape[0]
        rescaled_image = image[start:end, start:end, :]
        rescaled_image = np.squeeze(rescaled_image)
    else:
        start = int((output_shape[0] - target_shape[0]) / 2.0)
        end = start + target_shape[0]
        rescaled_image = np.ones(output_shape) * fill_value
        rescaled_image = np.expand_dims(rescaled_image, axis=2)
        rescaled_image[start:end, start:end, :] = image
        rescaled_image = np.squeeze(rescaled_image)
    return rescaled_image


def rescale_vflow(rgb, mag, agl, scale, factor):
    rescaled_rgb = rescale(rgb, factor, fill_value=0)
    rescaled_agl = rescale(agl, factor, fill_value=np.nan)
    rescaled_mag = rescale(mag, factor, fill_value=np.nan)
    rescaled_mag[np.isfinite(rescaled_mag)] /= factor
    scale /= factor
    return rescaled_rgb, rescaled_mag, rescaled_agl, scale


def warp_flow(img, flow):
    cols, rows = flow.shape[:2]
    wflow = -np.copy(flow)
    wflow[:, :, 0] += np.arange(cols)
    wflow[:, :, 1] += np.arange(rows)[:, np.newaxis]
    res = cv2.remap(img, wflow, None, cv2.INTER_LINEAR)
    return res


def warp_agl(rgb, mag, angle, agl, scale_factor, max_scale_factor):
    mag = cv2.medianBlur(mag, 5)
    mag2 = mag * (scale_factor - 1.0)
    x2 = -mag2 * np.sin(angle)
    y2 = -mag2 * np.cos(angle)
    x2 = (x2 + 0.5).astype(np.int32)
    y2 = (y2 + 0.5).astype(np.int32)
    flow = np.stack([x2, y2], axis=2)
    flow, mask = invert_flow(flow, mag, 1.0 / max_scale_factor)
    flow = flow.astype(np.float32)
    flow = cv2.medianBlur(flow, 5)
    rgb = warp_flow(rgb, flow).astype(np.uint8)
    rgb = cv2.blur(rgb, (3, 3))
    agl = warp_flow(agl, flow)
    agl = agl * scale_factor
    mag = warp_flow(mag, flow)
    mag = mag * scale_factor
    return rgb, mag, agl
