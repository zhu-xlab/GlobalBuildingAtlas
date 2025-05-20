import sys
import time
from functools import partial
import math
import random
import numpy as np
import scipy.spatial
from PIL import Image, ImageDraw, ImageFilter
import skimage.draw
import skimage
# from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
# from multiprocess import Pool
# import multiprocess
from tqdm import tqdm
import cv2
import pdb
import torch
import torch.nn.functional as F

from skimage.measure import approximate_polygon
from scipy.ndimage import distance_transform_edt, grey_dilation

import shapely.geometry
import shapely.affinity
import shapely.ops
import shapely.prepared
import shapely.validation
import shapely
import rasterio
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AffinityPropagation
from scipy.optimize import linear_sum_assignment


def is_polygon_clockwise(polygon):
    rolled_polygon = np.roll(polygon, shift=1, axis=0)
    double_signed_area = np.sum((rolled_polygon[:, 0] - polygon[:, 0]) * (rolled_polygon[:, 1] + polygon[:, 1]))
    if 0 < double_signed_area:
        return True
    else:
        return False


def orient_polygon(polygon, orientation="CW"):
    poly_is_orientated_cw = is_polygon_clockwise(polygon)
    if (poly_is_orientated_cw and orientation == "CCW") or (not poly_is_orientated_cw and orientation == "CW"):
        return np.flip(polygon, axis=0)
    else:
        return polygon


def orient_polygons(polygons, orientation="CW"):
    return [orient_polygon(polygon, orientation=orientation) for polygon in polygons]


def raster_to_polygon(image, vertex_count):
    contours = skimage.measure.find_contours(image, 0.5)
    contour = np.empty_like(contours[0])
    contour[:, 0] = contours[0][:, 1]
    contour[:, 1] = contours[0][:, 0]

    # Simplify until vertex_count
    tolerance = 0.1
    tolerance_step = 0.1
    simplified_contour = contour
    while 1 + vertex_count < len(simplified_contour):
        simplified_contour = approximate_polygon(contour, tolerance=tolerance)
        tolerance += tolerance_step

    simplified_contour = simplified_contour[:-1]

    # plt.imshow(image, cmap="gray")
    # plot_polygon(simplified_contour, draw_labels=False)
    # plt.show()

    return simplified_contour


def l2diffs(polygon1, polygon2):
    """
    Computes vertex-wise L2 difference between the two polygons.
    As the two polygons may not have the same starting vertex,
    all shifts are considred and the shift resulting in the minimum mean L2 difference is chosen
    
    :param polygon1: 
    :param polygon2: 
    :return: 
    """
    # Make polygons of equal length
    if len(polygon1) != len(polygon2):
        while len(polygon1) < len(polygon2):
            polygon1 = np.append(polygon1, [polygon1[-1, :]], axis=0)
        while len(polygon2) < len(polygon1):
            polygon2 = np.append(polygon2, [polygon2[-1, :]], axis=0)
    vertex_count = len(polygon1)

    def naive_l2diffs(polygon1, polygon2):
        naive_l2diffs_result = np.sqrt(np.power(np.sum(polygon1 - polygon2, axis=1), 2))
        return naive_l2diffs_result

    min_l2_diffs = naive_l2diffs(polygon1, polygon2)
    min_mean_l2_diffs = np.mean(min_l2_diffs, axis=0)
    for i in range(1, vertex_count):
        current_naive_l2diffs = naive_l2diffs(np.roll(polygon1, shift=i, axis=0), polygon2)
        current_naive_mean_l2diffs = np.mean(current_naive_l2diffs, axis=0)
        if current_naive_mean_l2diffs < min_mean_l2_diffs:
            min_l2_diffs = current_naive_l2diffs
            min_mean_l2_diffs = current_naive_mean_l2diffs
    return min_l2_diffs


def intersect_polygons(simple_polygon, multi_polygon):
    """

    :param input_polygon:
    :param target_polygon:
    :return: List of a simple polygon: [poly1, poly2,...] with a multi polygon: [[(x1, y1), (x2, y2), ...], [...]]
    """
    poly1 = shapely.geometry.Polygon(simple_polygon).buffer(0)
    poly2 = shapely.geometry.MultiPolygon(shapely.geometry.Polygon(polygon) for polygon in multi_polygon).buffer(0)
    intersection_poly = poly1.intersection(poly2)
    if 0 < intersection_poly.area:
        if intersection_poly.type == 'Polygon':
            coords = intersection_poly.exterior.coords
            return [coords]
        elif intersection_poly.type == 'MultiPolygon':
            ret_coords = []
            for poly in intersection_poly:
                coords = poly.exterior.coords
                ret_coords.append(coords)
            return ret_coords
    return None


def check_intersection_with_polygon(input_polygon, target_polygon):
    poly1 = shapely.geometry.Polygon(input_polygon).buffer(0)
    poly2 = shapely.geometry.Polygon(target_polygon).buffer(0)
    intersection_poly = poly1.intersection(poly2)
    intersection_area = intersection_poly.area
    is_intersection = 0 < intersection_area
    return is_intersection


def check_intersection_with_polygons(input_polygon, target_polygons):
    """
    Returns True if there is an intersection with at least one polygon in target_polygons
    :param input_polygon:
    :param target_polygons:
    :return:
    """
    for target_polygon in target_polygons:
        if check_intersection_with_polygon(input_polygon, target_polygon):
            return True
    return False


def polygon_area(polygon):
    poly = shapely.geometry.Polygon(polygon).buffer(0)
    return poly.area


def polygon_union(polygon1, polygon2):
    poly1 = shapely.geometry.Polygon(polygon1).buffer(0)
    poly2 = shapely.geometry.Polygon(polygon2).buffer(0)
    union_poly = poly1.union(poly2)
    return np.array(union_poly.exterior.coords)


def polygon_iou(polygon1, polygon2):
    poly1 = shapely.geometry.Polygon(polygon1).buffer(0)
    poly2 = shapely.geometry.Polygon(polygon2).buffer(0)
    intersection_poly = poly1.intersection(poly2)
    union_poly = poly1.union(poly2)
    intersection_area = intersection_poly.area
    union_area = union_poly.area
    if union_area:
        iou = intersection_area / union_area
    else:
        iou = 0
    return iou


def generate_polygon(cx, cy, ave_radius, irregularity, spikeyness, vertex_count):
    """
    Start with the centre of the polygon at cx, cy,
    then creates the polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    cx, cy - coordinates of the "centre" of the polygon
    ave_radius - in px, the average radius of this polygon, this roughly controls how large the polygon is,
        really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to
        [0, 2 * pi / vertex_count]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius ave_radius.
        [0,1] will map to [0, ave_radius]
    vertex_count - self-explanatory

    Returns a list of vertices, in CCW order.
    """

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / vertex_count
    spikeyness = clip(spikeyness, 0, 1) * ave_radius

    # generate n angle steps
    angle_steps = []
    lower = (2 * math.pi / vertex_count) - irregularity
    upper = (2 * math.pi / vertex_count) + irregularity
    angle_sum = 0
    for i in range(vertex_count):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        angle_sum = angle_sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = angle_sum / (2 * math.pi)
    for i in range(vertex_count):
        angle_steps[i] = angle_steps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(vertex_count):
        r_i = clip(random.gauss(ave_radius, spikeyness), 0, 2 * ave_radius)
        x = cx + r_i * math.cos(angle)
        y = cy + r_i * math.sin(angle)
        points.append((x, y))

        angle = angle + angle_steps[i]

    return points


def clip(x, mini, maxi):
    if mini > maxi:
        return x
    elif x < mini:
        return mini
    elif x > maxi:
        return maxi
    else:
        return x


def scale_bounding_box(bounding_box, scale):
    half_width = math.ceil((bounding_box[2] - bounding_box[0]) * scale / 2)
    half_height = math.ceil((bounding_box[3] - bounding_box[1]) * scale / 2)
    center = [round((bounding_box[0] + bounding_box[2]) / 2), round((bounding_box[1] + bounding_box[3]) / 2)]
    scaled_bounding_box = [int(center[0] - half_width), int(center[1] - half_height), int(center[0] + half_width),
                           int(center[1] + half_height)]
    return scaled_bounding_box


def pad_bounding_box(bbox, pad):
    return [bbox[0] + pad, bbox[1] + pad, bbox[2] - pad, bbox[3] - pad]


def compute_bounding_box(polygon, scale=1, boundingbox_margin=0, fit=None):
    # Compute base bounding box
    bounding_box = [np.min(polygon[:, 0]), np.min(polygon[:, 1]), np.max(polygon[:, 0]), np.max(polygon[:, 1])]
    # Scale
    half_width = math.ceil((bounding_box[2] - bounding_box[0]) * scale / 2)
    half_height = math.ceil((bounding_box[3] - bounding_box[1]) * scale / 2)
    # Add margin
    half_width += boundingbox_margin
    half_height += boundingbox_margin
    # Compute square bounding box
    if fit == "square":
        half_width = half_height = max(half_width, half_height)
    center = [round((bounding_box[0] + bounding_box[2]) / 2), round((bounding_box[1] + bounding_box[3]) / 2)]
    bounding_box = [int(center[0] - half_width), int(center[1] - half_height), int(center[0] + half_width),
                    int(center[1] + half_height)]
    return bounding_box


def compute_patch(polygon, patch_size):
    centroid = np.mean(polygon, axis=0)
    half_height = half_width = patch_size / 2
    bounding_box = [math.ceil(centroid[0] - half_width), math.ceil(centroid[1] - half_height),
                    math.ceil(centroid[0] + half_width), math.ceil(centroid[1] + half_height)]
    return bounding_box


def bounding_box_within_bounds(bounding_box, bounds):
    return bounds[0] <= bounding_box[0] and bounds[1] <= bounding_box[1] and bounding_box[2] <= bounds[2] and \
           bounding_box[3] <= bounds[3]


def vertex_within_bounds(vertex, bounds):
    return bounds[0] <= vertex[0] <= bounds[2] and \
           bounds[1] <= vertex[1] <= bounds[3]


def edge_within_bounds(edge, bounds):
    return vertex_within_bounds(edge[0], bounds) and vertex_within_bounds(edge[1], bounds)


def bounding_box_area(bounding_box):
    return (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])


def convert_to_image_patch_space(polygon_image_space, bounding_box):
    polygon_image_patch_space = np.empty_like(polygon_image_space)
    polygon_image_patch_space[:, 0] = polygon_image_space[:, 0] - bounding_box[0]
    polygon_image_patch_space[:, 1] = polygon_image_space[:, 1] - bounding_box[1]
    return polygon_image_patch_space


def translate_polygons(polygons, translation):
    for polygon in polygons:
        polygon[:, 0] += translation[0]
        polygon[:, 1] += translation[1]
    return polygons


def strip_redundant_vertex(vertices, epsilon=1):
    assert len(vertices.shape) == 2  # Is a polygon
    new_vertices = vertices
    if 1 < vertices.shape[0]:
        if np.sum(np.absolute(vertices[0, :] - vertices[-1, :])) < epsilon:
            new_vertices = vertices[:-1, :]
    return new_vertices


def remove_doubles(vertices, epsilon=0.1):
    dists = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=-1)
    new_vertices = vertices[epsilon < dists]
    return new_vertices


def simplify_polygon(polygon, tolerance=1):
    approx_polygon = approximate_polygon(polygon, tolerance=tolerance)
    return approx_polygon


def simplify_polygons(polygons, tolerance=1):
    approx_polygons = []
    for polygon in polygons:
        approx_polygon = approximate_polygon(polygon, tolerance=tolerance)
        approx_polygons.append(approx_polygon)
    return approx_polygons


def pad_polygon(vertices, target_length):
    assert len(vertices.shape) == 2  # Is a polygon
    assert vertices.shape[0] <= target_length
    padding_length = target_length - vertices.shape[0]
    padding = np.tile(vertices[-1], [padding_length, 1])
    padded_vertices = np.append(vertices, padding, axis=0)
    return padded_vertices


def compute_diameter(polygon):
    dist = scipy.spatial.distance.cdist(polygon, polygon)
    return dist.max()


def plot_polygon(polygon, color=None, draw_labels=True, label_direction=1, indexing="xy", axis=None):
    import matplotlib.pyplot as plt

    if axis is None:
        axis = plt.gca()

    polygon_closed = np.append(polygon, [polygon[0, :]], axis=0)
    if indexing == "xy=":
        axis.plot(polygon_closed[:, 0], polygon_closed[:, 1], color=color, linewidth=3.0)
    elif indexing == "ij":
        axis.plot(polygon_closed[:, 1], polygon_closed[:, 0], color=color, linewidth=3.0)
    else:
        print("WARNING: Invalid indexing argument")

    if draw_labels:
        labels = range(1, polygon.shape[0] + 1)
        for label, x, y in zip(labels, polygon[:, 0], polygon[:, 1]):
            axis.annotate(
                label,
                xy=(x, y), xytext=(-20 * label_direction, 20 * label_direction),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc=color, alpha=0.75),
                arrowprops=dict(arrowstyle='->', color=color, connectionstyle='arc3,rad=0'))


def plot_polygons(polygons, color=None, draw_labels=True, label_direction=1, indexing="xy", axis=None):
    for polygon in polygons:
        plot_polygon(polygon, color=color, draw_labels=draw_labels, label_direction=label_direction, indexing=indexing,
                     axis=axis)


def compute_edge_normal(edge):
    normal = np.array([- (edge[1][1] - edge[0][1]),
                       edge[1][0] - edge[0][0]])
    normal_norm = np.sqrt(np.sum(np.square(normal)))
    normal /= normal_norm
    return normal


def compute_vector_angle(x, y):
    if x < 0.0:
        slope = y / x
        angle = np.pi + np.arctan(slope)
    elif 0.0 < x:
        slope = y / x
        angle = np.arctan(slope)
    else:
        if 0 < y:
            angle = np.pi / 2
        else:
            angle = 3 * np.pi / 2
    if angle < 0.0:
        angle += 2 * np.pi
    return angle


def compute_edge_normal_angle_edge(edge):
    normal = compute_edge_normal(edge)
    normal_x = normal[1]
    normal_y = normal[0]
    angle = compute_vector_angle(normal_x, normal_y)
    return angle


def polygon_in_bounding_box(polygon, bounding_box):
    """
    Returns True if all vertices of polygons are inside bounding_box
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    result = np.all(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[3])
        )
    )
    return result


def filter_polygons_in_bounding_box(polygons, bounding_box):
    """
    Only keep polygons that are fully inside bounding_box

    :param polygons: [shape(N, 2), ...]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    filtered_polygons = []
    for polygon in polygons:
        if polygon_in_bounding_box(polygon, bounding_box):
            filtered_polygons.append(polygon)
    return filtered_polygons


def transform_polygon_to_bounding_box_space(polygon, bounding_box):
    """

    :param polygon: shape(N, 2)
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    assert len(polygon.shape) and polygon.shape[1] == 2, "polygon should have shape (N, 2), not shape {}".format(
        polygon.shape)
    assert len(bounding_box) == 4, "bounding_box should have 4 elements: [row_min, col_min, row_max, col_max]"
    transformed_polygon = polygon.copy()
    transformed_polygon[:, 0] -= bounding_box[0]
    transformed_polygon[:, 1] -= bounding_box[1]
    return transformed_polygon


def transform_polygons_to_bounding_box_space(polygons, bounding_box):
    transformed_polygons = []
    for polygon in polygons:
        transformed_polygons.append(transform_polygon_to_bounding_box_space(polygon, bounding_box))
    return transformed_polygons


def crop_polygon_to_patch(polygon, bounding_box):
    return transform_polygon_to_bounding_box_space(polygon, bounding_box)


def crop_polygon_to_patch_if_touch(polygon, bounding_box):
    assert type(polygon) == np.ndarray, "polygon should be a numpy array, not {}".format(type(polygon))
    assert len(polygon.shape) == 2 and polygon.shape[1] == 2, "polygon should be of shape (N, 2), not {}".format(
        polygon.shape)
    # Verify that at least one vertex is inside bounding_box
    polygon_touches_patch = np.any(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[3])
        )
    )
    if polygon_touches_patch:
        return crop_polygon_to_patch(polygon, bounding_box)
    else:
        return None


def crop_polygons_to_patch_if_touch(polygons, bounding_box, return_indices=False):
    assert type(polygons) == list, "polygons should be a list"
    if return_indices:
        indices = []
    cropped_polygons = []
    for i, polygon in enumerate(polygons):
        cropped_polygon = crop_polygon_to_patch_if_touch(polygon, bounding_box)
        if cropped_polygon is not None:
            cropped_polygons.append(cropped_polygon)
            if return_indices:
                indices.append(i)
    if return_indices:
        return cropped_polygons, indices
    else:
        return cropped_polygons


def crop_polygons_to_patch(polygons, bounding_box):
    cropped_polygons = []
    for polygon in polygons:
        cropped_polygon = crop_polygon_to_patch(polygon, bounding_box)
        if cropped_polygon is not None:
            cropped_polygons.append(cropped_polygon)
    return cropped_polygons


def patch_polygons(polygons, minx, miny, maxx, maxy):
    """
    Filters out polygons that do not touch the bbox and translate those that do to the box's coordinate system.

    @param polygons: [shapely.geometry.Polygon, ...]
    @param maxy:
    @param maxx:
    @param miny:
    @param minx:
    @return: [shapely.geometry.Polygon, ...]
    """
    assert type(polygons) == list, "polygons should be a list"
    if len(polygons) == 0:
        return polygons
    assert type(polygons[0]) == shapely.geometry.Polygon, \
        f"Items of the polygons list should be of type shapely.geometry.Polygon, not {type(polygons[0])}"

    box_polygon = shapely.geometry.box(minx, miny, maxx, maxy)
    polygons = filter(box_polygon.intersects, polygons)

    polygons = map(partial(shapely.affinity.translate, xoff=-minx, yoff=-miny), polygons)

    return list(polygons)


def polygon_remove_holes(polygon):
    polygon_no_holes = []
    for coords in polygon:
        if not np.isnan(coords[0]) and not np.isnan(coords[1]):
            polygon_no_holes.append(coords)
        else:
            break
    return np.array(polygon_no_holes)


def polygons_remove_holes(polygons):
    gt_polygons_no_holes = []
    for polygon in polygons:
        gt_polygons_no_holes.append(polygon_remove_holes(polygon))
    return gt_polygons_no_holes


def apply_batch_disp_map_to_polygons(pred_disp_field_map_batch, disp_polygons_batch):
    """

    :param pred_disp_field_map_batch: shape(batch_size, height, width, 2)
    :param disp_polygons_batch: shape(batch_size, polygon_count, vertex_count, 2)
    :return:
    """

    # Apply all displacements at once
    batch_count = pred_disp_field_map_batch.shape[0]
    row_count = pred_disp_field_map_batch.shape[1]
    col_count = pred_disp_field_map_batch.shape[2]

    disp_polygons_batch_int = np.round(disp_polygons_batch).astype(np.int)
    # Clip coordinates to the field map:
    disp_polygons_batch_int_nearest_valid_field = np.maximum(0, disp_polygons_batch_int)
    disp_polygons_batch_int_nearest_valid_field[:, :, :, 0] = np.minimum(
        disp_polygons_batch_int_nearest_valid_field[:, :, :, 0], row_count - 1)
    disp_polygons_batch_int_nearest_valid_field[:, :, :, 1] = np.minimum(
        disp_polygons_batch_int_nearest_valid_field[:, :, :, 1], col_count - 1)

    aligned_disp_polygons_batch = disp_polygons_batch.copy()
    for batch_index in range(batch_count):
        mask = ~np.isnan(disp_polygons_batch[batch_index, :, :, 0])  # Checking one coordinate is enough
        aligned_disp_polygons_batch[batch_index, mask, 0] += pred_disp_field_map_batch[batch_index,
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 0],
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 1], 0].flatten()
        aligned_disp_polygons_batch[batch_index, mask, 1] += pred_disp_field_map_batch[batch_index,
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 0],
                                                                                       disp_polygons_batch_int_nearest_valid_field[
                                                                                           batch_index, mask, 1], 1].flatten()
    return aligned_disp_polygons_batch


def apply_disp_map_to_polygons(disp_field_map, polygons):
    """

    :param disp_field_map: shape(height, width, 2)
    :param polygon_list: [shape(N, 2), shape(M, 2), ...]
    :return:
    """
    disp_field_map_batch = np.expand_dims(disp_field_map, axis=0)
    disp_polygons = []
    for polygon in polygons:
        polygon_batch = np.expand_dims(np.expand_dims(polygon, axis=0), axis=0)  # Add batch and polygon_count dims
        disp_polygon_batch = apply_batch_disp_map_to_polygons(disp_field_map_batch, polygon_batch)
        disp_polygon_batch = disp_polygon_batch[0, 0]  # Remove batch and polygon_count dims
        disp_polygons.append(disp_polygon_batch)
    return disp_polygons


# This next function is somewhat redundant with apply_disp_map_to_polygons... (but displaces in the opposite direction)
def apply_displacement_field_to_polygons(polygons, disp_field_map):
    disp_polygons = []
    for polygon in polygons:
        mask_nans = np.isnan(polygon)  # Will be necessary when polygons with holes are handled
        polygon_int = np.round(polygon).astype(np.int)
        polygon_int_clipped = np.maximum(0, polygon_int)
        polygon_int_clipped[:, 0] = np.minimum(disp_field_map.shape[0] - 1, polygon_int_clipped[:, 0])
        polygon_int_clipped[:, 1] = np.minimum(disp_field_map.shape[1] - 1, polygon_int_clipped[:, 1])
        disp_polygon = polygon.copy()
        disp_polygon[~mask_nans[:, 0], 0] -= disp_field_map[polygon_int_clipped[~mask_nans[:, 0], 0],
                                                            polygon_int_clipped[~mask_nans[:, 0], 1], 0]
        disp_polygon[~mask_nans[:, 1], 1] -= disp_field_map[polygon_int_clipped[~mask_nans[:, 1], 0],
                                                            polygon_int_clipped[~mask_nans[:, 1], 1], 1]
        disp_polygons.append(disp_polygon)
    return disp_polygons


def apply_displacement_fields_to_polygons(polygons, disp_field_maps):
    disp_field_map_count = disp_field_maps.shape[0]
    disp_polygons_list = []
    for i in range(disp_field_map_count):
        disp_polygons = apply_displacement_field_to_polygons(polygons, disp_field_maps[i, :, :, :])
        disp_polygons_list.append(disp_polygons)
    return disp_polygons_list


def draw_line(shape, line, width, blur_radius=0):
    im = Image.new("L", (shape[1], shape[0]))
    # im_px_access = im.load()
    draw = ImageDraw.Draw(im)
    vertex_list = []
    for coords in line:
        vertex = (coords[1], coords[0])
        vertex_list.append(vertex)
    draw.line(vertex_list, fill=255, width=width)
    if 0 < blur_radius:
        im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    array = np.array(im) / 255
    return array


def draw_triangle(shape, triangle, blur_radius=0):
    im = Image.new("L", (shape[1], shape[0]))
    # im_px_access = im.load()
    draw = ImageDraw.Draw(im)
    vertex_list = []
    for coords in triangle:
        vertex = (coords[1], coords[0])
        vertex_list.append(vertex)
    draw.polygon(vertex_list, fill=255)
    if 0 < blur_radius:
        im = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    array = np.array(im) / 255
    return array


def draw_polygon(polygon, shape, fill=True, edges=True, vertices=True, line_width=3):
    # TODO: handle holes in polygons
    im = Image.new("RGB", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    vertex_list = []
    for coords in polygon:
        vertex = (coords[1], coords[0])
        if not np.isnan(vertex[0]) and not np.isnan(vertex[1]):
            vertex_list.append(vertex)
        else:
            break
    if edges:
        draw.line(vertex_list, fill=(0, 255, 0), width=line_width)
    if fill:
        draw.polygon(vertex_list, fill=(255, 0, 0))
    if vertices:
        draw.point(vertex_list, fill=(0, 0, 255))

    # Convert image to numpy array with the right number of channels
    array = np.array(im)
    selection = [fill, edges, vertices]
    selected_array = array[:, :, selection]
    return selected_array


def _draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius,
                  center[1] - radius,
                  center[0] + radius,
                  center[1] + radius], fill=fill, outline=None)


def draw_polygons(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    # TODO: handle holes in polygons
    polygons = polygons_remove_holes(polygons)
    polygons = polygons_close(polygons)

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
    else:
        draw_shape = shape
    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (draw_shape[1], draw_shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        if antialiasing:
            polygon *= 2
        vertex_list = []
        for coords in polygon:
            vertex_list.append((coords[1], coords[0]))
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(vertex_list, fill=255)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(vertex_list, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            for vertex in vertex_list:
                _draw_circle(draw, vertex, line_width / 2, fill=255)

    im_list = []
    if antialiasing:
        # resize images:
        for im_draw in im_draw_list:
            resize_shape = (shape[1], shape[0])
            im_list.append(im_draw[0].resize(resize_shape, Image.BILINEAR))
    else:
        for im_draw in im_draw_list:
            im_list.append(im_draw[0])

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im) for im in im_list]
    array = np.stack(array_list, axis=-1)
    return array


def draw_polygon_map(polygons, shape, fill=True, edges=True, vertices=True, line_width=3):
    """
    Alias for draw_polygon function

    :param polygons:
    :param shape:
    :param fill:
    :param edges:
    :param vertices:
    :param line_width:
    :return:
    """
    return draw_polygons(polygons, shape, fill=fill, edges=edges, vertices=vertices, line_width=line_width)


def draw_polygon_maps(polygons_list, shape, fill=True, edges=True, vertices=True, line_width=3):
    polygon_maps_list = []
    for polygons in polygons_list:
        polygon_map = draw_polygon_map(polygons, shape, fill=fill, edges=edges, vertices=vertices,
                                       line_width=line_width)
        polygon_maps_list.append(polygon_map)
    disp_field_maps = np.stack(polygon_maps_list, axis=0)
    return disp_field_maps


def swap_coords(polygon):
    polygon_new = polygon.copy()
    polygon_new[..., 0] = polygon[..., 1]
    polygon_new[..., 1] = polygon[..., 0]
    return polygon_new


def prepare_polygons_for_tfrecord(gt_polygons, disp_polygons_list, boundingbox=None):
    assert len(gt_polygons)

    # print("Starting to crop polygons")
    # start = time.time()

    dtype = gt_polygons[0].dtype
    cropped_gt_polygons = []
    cropped_disp_polygons_list = [[] for i in range(len(disp_polygons_list))]
    polygon_length = 0
    for polygon_index, gt_polygon in enumerate(gt_polygons):
        if boundingbox is not None:
            cropped_gt_polygon = crop_polygon_to_patch_if_touch(gt_polygon, boundingbox)
        else:
            cropped_gt_polygon = gt_polygon
        if cropped_gt_polygon is not None:
            cropped_gt_polygons.append(cropped_gt_polygon)
            if polygon_length < cropped_gt_polygon.shape[0]:
                polygon_length = cropped_gt_polygon.shape[0]
            # Crop disp polygons
            for disp_index, disp_polygons in enumerate(disp_polygons_list):
                disp_polygon = disp_polygons[polygon_index]
                if boundingbox is not None:
                    cropped_disp_polygon = crop_polygon_to_patch(disp_polygon, boundingbox)
                else:
                    cropped_disp_polygon = disp_polygon
                cropped_disp_polygons_list[disp_index].append(cropped_disp_polygon)

    # end = time.time()
    # print("Finished cropping polygons in in {}s".format(end - start))
    #
    # print("Starting to pad polygons")
    # start = time.time()

    polygon_count = len(cropped_gt_polygons)
    if polygon_count:
        # Add +1 to both dimensions for end-of-item NaNs
        padded_gt_polygons = np.empty((polygon_count + 1, polygon_length + 1, 2), dtype=dtype)
        padded_gt_polygons[:, :, :] = np.nan
        padded_disp_polygons_array = np.empty((len(disp_polygons_list), polygon_count + 1, polygon_length + 1, 2),
                                              dtype=dtype)
        padded_disp_polygons_array[:, :, :] = np.nan
        for i, polygon in enumerate(cropped_gt_polygons):
            padded_gt_polygons[i, 0:polygon.shape[0], :] = polygon
        for j, polygons in enumerate(cropped_disp_polygons_list):
            for i, polygon in enumerate(polygons):
                padded_disp_polygons_array[j, i, 0:polygon.shape[0], :] = polygon
    else:
        padded_gt_polygons = padded_disp_polygons_array = None

    # end = time.time()
    # print("Finished padding polygons in in {}s".format(end - start))

    return padded_gt_polygons, padded_disp_polygons_array


def prepare_stages_polygons_for_tfrecord(gt_polygons, disp_polygons_list_list, boundingbox):
    assert len(gt_polygons)

    print(gt_polygons)
    print(disp_polygons_list_list)

    exit()

    # print("Starting to crop polygons")
    # start = time.time()

    dtype = gt_polygons[0].dtype
    cropped_gt_polygons = []
    cropped_disp_polygons_list_list = [[[] for i in range(len(disp_polygons_list))] for disp_polygons_list in
                                       disp_polygons_list_list]
    polygon_length = 0
    for polygon_index, gt_polygon in enumerate(gt_polygons):
        cropped_gt_polygon = crop_polygon_to_patch_if_touch(gt_polygon, boundingbox)
        if cropped_gt_polygon is not None:
            cropped_gt_polygons.append(cropped_gt_polygon)
            if polygon_length < cropped_gt_polygon.shape[0]:
                polygon_length = cropped_gt_polygon.shape[0]
            # Crop disp polygons
            for stage_index, disp_polygons_list in enumerate(disp_polygons_list_list):
                for disp_index, disp_polygons in enumerate(disp_polygons_list):
                    disp_polygon = disp_polygons[polygon_index]
                    cropped_disp_polygon = crop_polygon_to_patch(disp_polygon, boundingbox)
                    cropped_disp_polygons_list_list[stage_index][disp_index].append(cropped_disp_polygon)

    # end = time.time()
    # print("Finished cropping polygons in in {}s".format(end - start))
    #
    # print("Starting to pad polygons")
    # start = time.time()

    polygon_count = len(cropped_gt_polygons)
    if polygon_count:
        # Add +1 to both dimensions for end-of-item NaNs
        padded_gt_polygons = np.empty((polygon_count + 1, polygon_length + 1, 2), dtype=dtype)
        padded_gt_polygons[:, :, :] = np.nan
        padded_disp_polygons_array = np.empty(
            (len(disp_polygons_list_list), len(disp_polygons_list_list[0]), polygon_count + 1, polygon_length + 1, 2),
            dtype=dtype)
        padded_disp_polygons_array[:, :, :] = np.nan
        for i, polygon in enumerate(cropped_gt_polygons):
            padded_gt_polygons[i, 0:polygon.shape[0], :] = polygon
        for k, cropped_disp_polygons_list in enumerate(cropped_disp_polygons_list_list):
            for j, polygons in enumerate(cropped_disp_polygons_list):
                for i, polygon in enumerate(polygons):
                    padded_disp_polygons_array[k, j, i, 0:polygon.shape[0], :] = polygon
    else:
        padded_gt_polygons = padded_disp_polygons_array = None

    # end = time.time()
    # print("Finished padding polygons in in {}s".format(end - start))

    return padded_gt_polygons, padded_disp_polygons_array


def rescale_polygon(polygons, scaling_factor):
    """

    :param polygons:
    :return: scaling_factor
    """
    if len(polygons):
        rescaled_polygons = [polygon * scaling_factor for polygon in polygons]
        return rescaled_polygons
    else:
        return polygons


def get_edge_center(edge):
    return np.mean(edge, axis=0)


def get_edge_length(edge):
    return np.sqrt(np.sum(np.square(edge[0] - edge[1])))


def get_edges_angle(edge1, edge2):
    x1 = edge1[1, 0] - edge1[0, 0]
    y1 = edge1[1, 1] - edge1[0, 1]
    x2 = edge2[1, 0] - edge2[0, 0]
    y2 = edge2[1, 1] - edge2[0, 1]
    angle1 = compute_vector_angle(x1, y1)
    angle2 = compute_vector_angle(x2, y2)
    edges_angle = math.fabs(angle1 - angle2) % (2 * math.pi)
    if math.pi < edges_angle:
        edges_angle = 2 * math.pi - edges_angle
    return edges_angle


def compute_angle_two_points(point_source, point_target):
    vector = point_target - point_source
    angle = compute_vector_angle(vector[0], vector[1])
    return angle


def compute_angle_three_points(point_source, point_target1, point_target2):
    squared_dist_source_target1 = math.pow((point_source[0] - point_target1[0]), 2) + math.pow(
        (point_source[1] - point_target1[1]), 2)
    squared_dist_source_target2 = math.pow((point_source[0] - point_target2[0]), 2) + math.pow(
        (point_source[1] - point_target2[1]), 2)
    squared_dist_target1_target2 = math.pow((point_target1[0] - point_target2[0]), 2) + math.pow(
        (point_target1[1] - point_target2[1]), 2)
    dist_source_target1 = math.sqrt(squared_dist_source_target1)
    dist_source_target2 = math.sqrt(squared_dist_source_target2)
    try:
        cos = (squared_dist_source_target1 + squared_dist_source_target2 - squared_dist_target1_target2) / (
                2 * dist_source_target1 * dist_source_target2)
    except ZeroDivisionError:
        return float('inf')
    cos = max(min(cos, 1),
              -1)  # Avoid some math domain error due to cos being slightly bigger than 1 (from floating point operations)
    angle = math.acos(cos)
    return angle


def are_edges_overlapping(edge1, edge2, threshold):
    """
    Checks if at least 2 different vertices of either edge lies on the other edge: it characterizes an overlap
    :param edge1:
    :param edge2:
    :param threshold:
    :return:
    """
    count_list = [
        is_vertex_on_edge(edge1[0], edge2, threshold),
        is_vertex_on_edge(edge1[1], edge2, threshold),
        is_vertex_on_edge(edge2[0], edge1, threshold),
        is_vertex_on_edge(edge2[1], edge1, threshold),
    ]
    # Count number of identical vertices
    identical_vertex_list = [
        np.array_equal(edge1[0], edge2[0]),
        np.array_equal(edge1[0], edge2[1]),
        np.array_equal(edge1[1], edge2[0]),
        np.array_equal(edge1[1], edge2[1]),
    ]
    adjusted_count = np.sum(count_list) - np.sum(identical_vertex_list)
    return 2 <= adjusted_count


# def are_edges_collinear(edge1, edge2, angle_threshold):
#     edges_angle = get_edges_angle(edge1, edge2)
#     return edges_angle < angle_threshold


def get_line_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z


def are_edges_intersecting(edge1, edge2, epsilon=1e-6):
    """
    edge1 and edge2 should not have a common vertex between them
    :param edge1:
    :param edge2:
    :return:
    """
    intersect = get_line_intersect(edge1[0], edge1[1], edge2[0], edge2[1])
    # print("---")
    # print(edge1)
    # print(edge2)
    # print(intersect)
    if intersect[0] == float('inf') or intersect[1] == float('inf'):
        # Lines don't intersect
        return False
    else:
        # Lines intersect
        # Check if intersect point belongs to both edges
        angle1 = compute_angle_three_points(intersect, edge1[0], edge1[1])
        angle2 = compute_angle_three_points(intersect, edge2[0], edge2[1])
        intersect_belongs_to_edges = (math.pi - epsilon) < angle1 and (math.pi - epsilon) < angle2
        return intersect_belongs_to_edges


def shorten_edge(edge, length_to_cut1, length_to_cut2, min_length):
    center = get_edge_center(edge)
    total_length = get_edge_length(edge)
    new_length = total_length - length_to_cut1 - length_to_cut2
    if min_length <= new_length:
        scale = new_length / total_length
        new_edge = (edge.copy() - center) * scale + center
        return new_edge
    else:
        return None


def is_edge_in_triangle(edge, triangle):
    return edge[0] in triangle and edge[1] in triangle


def get_connectivity_of_edge(edge, triangles):
    connectivity = 0
    for triangle in triangles:
        connectivity += is_edge_in_triangle(edge, triangle)
    return connectivity


def get_connectivity_of_edges(edges, triangles):
    connectivity_of_edges = []
    for edge in edges:
        connectivity_of_edge = get_connectivity_of_edge(edge, triangles)
        connectivity_of_edges.append(connectivity_of_edge)
    return connectivity_of_edges


def polygon_to_closest_int(polygons):
    int_polygons = []
    for polygon in polygons:
        int_polygon = np.round(polygon)
        int_polygons.append(int_polygon)
    return int_polygons


def is_vertex_on_edge(vertex, edge, threshold):
    """
    :param vertex:
    :param edge:
    :param threshold:
    :return:
    """
    # Compare distances sum to edge length
    edge_length = get_edge_length(edge)
    dist1 = get_edge_length([vertex, edge[0]])
    dist2 = get_edge_length([vertex, edge[1]])
    vertex_on_edge = (dist1 + dist2) < (edge_length + threshold)
    return vertex_on_edge


def get_face_edges(face_vertices):
    edges = []
    prev_vertex = face_vertices[0]
    for vertex in face_vertices[1:]:
        edge = (prev_vertex, vertex)
        edges.append(edge)

        # For next iteration:
        prev_vertex = vertex
    return edges


def find_edge_in_face(edge, face_vertices):
    # Copy inputs list so that we don't modify it
    face_vertices = face_vertices[:]
    face_vertices.append(face_vertices[0])  # Close face (does not matter if it is already closed)
    edges = get_face_edges(face_vertices)
    index = edges.index(edge)
    return index


def clean_degenerate_face_edges(face_vertices):
    def recursive_clean_degenerate_face_edges(open_face_vertices):
        face_vertex_count = len(open_face_vertices)
        cleaned_open_face_vertices = []
        skip = False
        for index in range(face_vertex_count):
            if skip:
                skip = False
            else:
                prev_vertex = open_face_vertices[(index - 1) % face_vertex_count]
                vertex = open_face_vertices[index]
                next_vertex = open_face_vertices[(index + 1) % face_vertex_count]
                if prev_vertex != next_vertex:
                    cleaned_open_face_vertices.append(vertex)
                else:
                    skip = True
        if len(cleaned_open_face_vertices) < face_vertex_count:
            return recursive_clean_degenerate_face_edges(cleaned_open_face_vertices)
        else:
            return cleaned_open_face_vertices

    open_face_vertices = face_vertices[:-1]
    cleaned_face_vertices = recursive_clean_degenerate_face_edges(open_face_vertices)
    # Close cleaned_face_vertices
    cleaned_face_vertices.append(cleaned_face_vertices[0])
    return cleaned_face_vertices


def merge_vertices(main_face_vertices, extra_face_vertices, common_edge):
    sorted_common_edge = tuple(sorted(common_edge))
    open_face_vertices_pair = (main_face_vertices[:-1], extra_face_vertices[:-1])
    face_index = 0  # 0: current_face == main_face, 1: current_face == extra_face
    vertex_index = 0
    start_vertex = vertex = open_face_vertices_pair[face_index][vertex_index]
    merged_face_vertices = [start_vertex]
    faces_merged = False
    while not faces_merged:
        # Get next vertex
        next_vertex_index = (vertex_index + 1) % len(open_face_vertices_pair[face_index])
        next_vertex = open_face_vertices_pair[face_index][next_vertex_index]
        edge = (vertex, next_vertex)
        sorted_edge = tuple(sorted(edge))
        if sorted_edge == sorted_common_edge:
            # Switch current face
            face_index = 1 - face_index
            # Find vertex_index in new current face
            reverse_edge = (edge[1], edge[0])  # Because we are now on the other face
            edge_index = find_edge_in_face(reverse_edge, open_face_vertices_pair[face_index])
            vertex_index = edge_index + 1  # Index of the second vertex of edge
            # vertex_index = open_face_vertices_pair[face_index].index(vertex)
        vertex_index = (vertex_index + 1) % len(open_face_vertices_pair[face_index])
        vertex = open_face_vertices_pair[face_index][vertex_index]
        merged_face_vertices.append(vertex)
        faces_merged = vertex == start_vertex  # This also makes the merged_face closed
    # Remove degenerate face edges (edges where the face if on both sides of it)
    cleaned_merged_face_vertices = clean_degenerate_face_edges(merged_face_vertices)
    return cleaned_merged_face_vertices


def polygon_close(polygon):
    return np.concatenate((polygon, polygon[0:1, :]), axis=0)


def polygons_close(polygons):
    return [polygon_close(polygon) for polygon in polygons]


# def init_cross_field(polygons, shape):
#     """
#     Cross field: {v_1, v_2, -v_1, -v_2} encoded as {v_1, v_2}.
#     This is not invariant to symmetries.
#
#     :param polygons:
#     :param shape:
#     :return: cross_field_array (shape[0], shape[1], 2), dtype=np.int8
#     """
#     def draw_edge(edge, v1):
#         rr, cc = skimage.draw.line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
#         mask = (0 <= rr) & (rr < shape[0]) & (0 <= cc) & (cc < shape[1])
#         cross_field_array[rr[mask], cc[mask], 0] = v1.real
#         cross_field_array[rr[mask], cc[mask], 1] = v1.imag
#
#     polygons = polygons_remove_holes(polygons)
#     polygons = polygons_close(polygons)
#
#     cross_field_array = np.zeros(shape + (4,), dtype=np.float)
#
#     for polygon in polygons:
#         # --- edges:
#         edge_vect_array = np.diff(polygon, axis=0)
#         norm = np.linalg.norm(edge_vect_array, axis=1, keepdims=True)
#         # if not np.all(0 < norm):
#         #     print("WARNING: one of the norms is zero, which cannot be used to divide")
#         #     print("polygon that raised this warning:")
#         #     print(polygon)
#         #     exit()
#         edge_dir_array = edge_vect_array / norm
#         edge_v1_array = edge_dir_array.view(np.complex)[..., 0]
#         # edge_v2_array is zero
#
#         # --- vertices:
#         vertex_v1_array = edge_v1_array
#         vertex_v2_array = - np.roll(edge_v1_array, 1, axis=0)
#
#         # --- Draw values
#         polygon = polygon.astype(np.int)
#
#         for i in range(polygon.shape[0] - 1):
#             edge = (polygon[i], polygon[i+1])
#             v1 = edge_v1_array[i]
#             draw_edge(edge, v1)
#
#         vertex_array = polygon[:-1]
#         mask = (0 <= vertex_array[:, 0]) & (vertex_array[:, 0] < shape[0])\
#                & (0 <= vertex_array[:, 1]) & (vertex_array[:, 1] < shape[1])
#         cross_field_array[vertex_array[mask, 0], vertex_array[mask, 1], 0] = vertex_v1_array[mask].real
#         cross_field_array[vertex_array[mask, 0], vertex_array[mask, 1], 1] = vertex_v1_array[mask].imag
#         cross_field_array[vertex_array[mask, 0], vertex_array[mask, 1], 2] = vertex_v2_array[mask].real
#         cross_field_array[vertex_array[mask, 0], vertex_array[mask, 1], 3] = vertex_v2_array[mask].imag
#
#     # --- Encode cross-field with integer complex to save memory because abs(cross_field_array) <= 1 anyway.
#     cross_field_array = (127*cross_field_array).astype(np.int8)
#
#     return cross_field_array


# def init_angle_field(polygons, shape):
#     """
#     Angle field {\theta_1} the tangent vector's angle for every pixel, specified on the polygon edges.
#     Angle between 0 and pi.
#     Also indices of those angle values.
#     This is not invariant to symmetries.
#
#     :param polygons:
#     :param shape:
#     :return: (angles: np.array((num_edge_pixels, ), dtype=np.uint8),
#               indices: np.array((num_edge_pixels, 2), dtype=np.int))
#     """
#     def draw_edge(edge, angle):
#         rr, cc = skimage.draw.line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
#         edge_mask = (0 <= rr) & (rr < shape[0]) & (0 <= cc) & (cc < shape[1])
#         angle_field_array[rr[edge_mask], cc[edge_mask]] = angle
#         mask[rr[edge_mask], cc[edge_mask]] = True
#
#     polygons = polygons_remove_holes(polygons)
#     polygons = polygons_close(polygons)
#
#     angle_field_array = np.zeros(shape, dtype=np.float)
#     mask = np.zeros(shape, dtype=np.bool)
#
#     for polygon in polygons:
#         # --- edges:
#         edge_vect_array = np.diff(polygon, axis=0)
#         edge_angle_array = np.angle(edge_vect_array[:, 0] + 1j * edge_vect_array[:, 1])
#         neg_indices = np.where(edge_angle_array < 0)
#         edge_angle_array[neg_indices] += np.pi
#
#         # --- Draw values
#         polygon = polygon.astype(np.int)
#
#         for i in range(polygon.shape[0] - 1):
#             edge = (polygon[i], polygon[i+1])
#             angle = edge_angle_array[i]
#             draw_edge(edge, angle)
#
#     # --- Encode angle-field with positive integers to save memory because angle is between 0 and pi.
#     indices = np.stack(np.where(mask), axis=-1)
#     angles = angle_field_array[indices[:, 0], indices[:, 1]]
#     angles = (255*angles/np.pi).round().astype(np.uint8)
#
#     return angles, indices


def init_angle_field(polygons, shape, line_width=1):
    """
    Angle field {\theta_1} the tangent vector's angle for every pixel, specified on the polygon edges.
    Angle between 0 and pi.
    This is not invariant to symmetries.

    :param polygons:
    :param shape:
    :return: (angles: np.array((num_edge_pixels, ), dtype=np.uint8),
              mask: np.array((num_edge_pixels, 2), dtype=np.int))
    """
    assert type(polygons) == list, "polygons should be a list"

    polygons = polygons_remove_holes(polygons)
    polygons = polygons_close(polygons)

    im = Image.new("L", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    for polygon in polygons:
        # --- edges:
        edge_vect_array = np.diff(polygon, axis=0)
        edge_angle_array = np.angle(edge_vect_array[:, 0] + 1j * edge_vect_array[:, 1])
        neg_indices = np.where(edge_angle_array < 0)
        edge_angle_array[neg_indices] += np.pi

        for i in range(polygon.shape[0] - 1):
            edge = (polygon[i], polygon[i + 1])
            angle = edge_angle_array[i]
            uint8_angle = int((255 * angle / np.pi).round())
            line = [(edge[0][1], edge[0][0]), (edge[1][1], edge[1][0])]
            draw.line(line, fill=uint8_angle, width=line_width)
            _draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)
        _draw_circle(draw, line[1], radius=line_width / 2, fill=uint8_angle)

    # Convert image to numpy array
    array = np.array(im)
    return array


def plot_geometries(axis, geometries, linewidths=1, markersize=3):
    if len(geometries):
        patches = []
        for i, geometry in enumerate(geometries):
            if geometry.geom_type == "Polygon":
                polygon = shapely.geometry.Polygon(geometry)
                if not polygon.is_empty:
                    patch = PolygonPatch(polygon)
                    patches.append(patch)
                axis.plot(*polygon.exterior.xy, marker="o", markersize=markersize)
                for interior in polygon.interiors:
                    axis.plot(*interior.xy, marker="o", markersize=markersize)
            elif geometry.geom_type == "LineString" or geometry.geom_type == "LinearRing":
                axis.plot(*geometry.xy, marker="o", markersize=markersize)
            else:
                raise NotImplementedError(f"Geom type {geometry.geom_type} not recognized.")
        random.seed(1)
        colors = random.choices([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0.5, 1, 0, 1],
            [1, 0.5, 0, 1],
            [0.5, 0, 1, 1],
            [1, 0, 0.5, 1],
            [0, 0.5, 1, 1],
            [0, 1, 0.5, 1],
        ], k=len(patches))
        edgecolors = np.array(colors)
        facecolors = edgecolors.copy()
        p = PatchCollection(patches, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths)
        axis.add_collection(p)


def sample_geometry(geom, density):
    """
    Sample edges of geom with a homogeneous density.

    @param geom:
    @param density:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        sampled_geom = shapely.geometry.GeometryCollection([sample_geometry(g, density) for g in geom.geoms])

        # toc = time.time()
        # print(f"sample_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        sampled_exterior = sample_geometry(geom.exterior, density)
        sampled_interiors = [sample_geometry(interior, density) for interior in geom.interiors]
        sampled_geom = shapely.geometry.Polygon(sampled_exterior, sampled_interiors)
    elif isinstance(geom, shapely.geometry.LineString):
        sampled_x = []
        sampled_y = []
        coords = np.array(geom.coords[:])
        lengths = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
        for i in range(len(lengths)):
            start = geom.coords[i]
            end = geom.coords[i + 1]
            length = lengths[i]
            num = max(1, int(round(length / density))) + 1
            x_seq = np.linspace(start[0], end[0], num)
            y_seq = np.linspace(start[1], end[1], num)
            if 0 < i:
                x_seq = x_seq[1:]
                y_seq = y_seq[1:]
            sampled_x.append(x_seq)
            sampled_y.append(y_seq)
        sampled_x = np.concatenate(sampled_x)
        sampled_y = np.concatenate(sampled_y)
        sampled_coords = zip(sampled_x, sampled_y)
        sampled_geom = shapely.geometry.LineString(sampled_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return sampled_geom

#
# def sample_half_tangent_endpoints(geom, length=0.1):
#     """
#     Add 2 vertices per edge, very close to the edge's endpoints. They represent both half-tangent endpoints
#     @param geom:
#     @param length:
#     @return:
#     """
#     if isinstance(geom, shapely.geometry.GeometryCollection):
#         sampled_geom = shapely.geometry.GeometryCollection([sample_half_tangent_endpoints(g, length) for g in geom])
#     elif isinstance(geom, shapely.geometry.Polygon):
#         sampled_exterior = sample_half_tangent_endpoints(geom.exterior, length)
#         sampled_interiors = [sample_half_tangent_endpoints(interior, length) for interior in geom.interiors]
#         sampled_geom = shapely.geometry.Polygon(sampled_exterior, sampled_interiors)
#     elif isinstance(geom, shapely.geometry.LineString):
#         coords = np.array(geom.coords[:])
#         edge_vecs = coords[1:] - coords[:-1]
#         norms = np.linalg.norm(edge_vecs, axis=1)
#         edge_dirs = edge_vecs / norms[:, None]
#         sampled_coords = [coords[0]]  # Init with first vertex
#         for edge_i in range(edge_dirs.shape[0]):
#             first_half_tangent_endpoint = coords[edge_i] + length * edge_dirs[edge_i]
#             sampled_coords.append(first_half_tangent_endpoint)
#             second_half_tangent_endpoint = coords[edge_i + 1] - length * edge_dirs[edge_i]
#             sampled_coords.append(second_half_tangent_endpoint)
#             sampled_coords.append(coords[edge_i + 1])  # Next vertex
#         sampled_geom = shapely.geometry.LineString(sampled_coords)
#     else:
#         raise TypeError(f"geom of type {type(geom)} not supported!")
#     return sampled_geom


def point_project_onto_geometry(coord, target):
    point = shapely.geometry.Point(coord)
    _, projected_point = shapely.ops.nearest_points(point, target)
    # dist = point.distance(projected_point)
    return projected_point.coords[0]


def project_onto_geometry(geom, target, pool):
    """
    Projects all points from line_string onto target.
    @param geom:
    @param target:
    @param pool:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        if pool is None:
            projected_geom = [project_onto_geometry(g, target, pool=pool) for g in geom.geoms]
        else:
            partial_project_onto_geometry = partial(project_onto_geometry, target=target)
            projected_geom = pool.map(partial_project_onto_geometry, geom)
        projected_geom = shapely.geometry.GeometryCollection(projected_geom)

        # toc = time.time()
        # print(f"project_onto_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        projected_exterior = project_onto_geometry(geom.exterior, target)
        projected_interiors = [project_onto_geometry(interior, target) for interior in geom.interiors]
        try:
            projected_geom = shapely.geometry.Polygon(projected_exterior, projected_interiors)
        except shapely.errors.TopologicalError as e:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [geom])
            plot_geometries(ax[1], target)
            plot_geometries(ax[2], [projected_exterior, *projected_interiors])
            fig.tight_layout()
            plt.show()
            raise e
    elif isinstance(geom, shapely.geometry.LineString):
        projected_coords = [point_project_onto_geometry(coord, target) for coord in geom.coords]
        projected_geom = shapely.geometry.LineString(projected_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return projected_geom

#
# def compute_edge_measures(geom1, geom2, max_stretch, metric_name="cosine"):
#     """
#
#     @param geom1:
#     @param geom2:
#     @param max_stretch: Edges of geom2 than are longer than those of geom1 with a factor greater than max_stretch are ignored
#     @param metric_name:
#     @return:
#     """
#     assert type(geom1) == type(geom2), f"geom1 and geom2 must be of the same type, not {type(geom1)} and {type(geom2)}"
#     if isinstance(geom1, shapely.geometry.GeometryCollection):
#         # tic = time.time()
#
#         edge_measures_edge_dists_list = [compute_edge_measures(_geom1, _geom2, max_stretch, metric_name=metric_name) for _geom1, _geom2 in zip(geom1, geom2)]
#         if len(edge_measures_edge_dists_list):
#             edge_measures_list, edge_dists_list = zip(*edge_measures_edge_dists_list)
#             edge_measures = np.concatenate(edge_measures_list)
#             edge_dists = np.concatenate(edge_dists_list)
#         else:
#             edge_measures = np.array([])
#             edge_dists = np.array([])
#
#         # toc = time.time()
#         # print(f"compute_edge_distance: {toc - tic}s")
#     # elif isinstance(geom1, shapely.geometry.Polygon):
#     #     distances_exterior = compute_edge_distance(geom1.exterior, geom2.exterior, tolerance, max_stretch, dist=dist)
#     #     distances_interiors = [compute_edge_distance(interior1, interior2, tolerance, max_stretch, dist=dist) for interior1, interior2 in zip(geom1.interiors, geom2.interiors)]
#     #     distances = [distances_exterior, *distances_interiors]
#     #     distances = np.concatenate(distances)
#     elif isinstance(geom1, shapely.geometry.LineString):
#         assert len(geom1.coords) == len(geom2.coords), "geom1 and geom2 must have the same length"
#         points1 = np.array(geom1.coords)
#         points2 = np.array(geom2.coords)
#         # Mark points that are farther away than tolerance between points1 and points2 to remove then from further computation
#         point_dists = np.linalg.norm(points1 - points2, axis=1)
#         if metric_name == "cosine":
#             edges1 = points1[1:] - points1[:-1]
#             edges2 = points2[1:] - points2[:-1]
#             edge_dists = (point_dists[1:] + point_dists[:-1]) / 2
#             # Remove edges with a norm of zero
#             norm1 = np.linalg.norm(edges1, axis=1)
#             norm2 = np.linalg.norm(edges2, axis=1)
#             norm_valid_mask = 0 < norm1 * norm2
#             edges1 = edges1[norm_valid_mask]
#             edges2 = edges2[norm_valid_mask]
#             norm1 = norm1[norm_valid_mask]
#             norm2 = norm2[norm_valid_mask]
#             edge_dists = edge_dists[norm_valid_mask]
#             # Remove edges that have been stretched more than max_stretch
#             stretch = norm2 / norm1
#             stretch_valid_mask = np.logical_and(1 / max_stretch < stretch, stretch < max_stretch)
#             edges1 = edges1[stretch_valid_mask]
#             edges2 = edges2[stretch_valid_mask]
#             norm1 = norm1[stretch_valid_mask]
#             norm2 = norm2[stretch_valid_mask]
#             edge_dists = edge_dists[stretch_valid_mask]
#             # Compute
#             edge_measures = np.sum(np.multiply(edges1, edges2), axis=1) / (norm1 * norm2)
#         else:
#             raise NotImplemented(f"Metric '{metric_name}' is not implemented")
#     else:
#         raise TypeError(f"geom of type {type(geom1)} not supported!")
#     return edge_measures, edge_dists


def compute_contour_measure(pred_polygon, gt_polygon, sampling_spacing, max_stretch, metric_name="cosine"):

    pred_contours = shapely.geometry.GeometryCollection([pred_polygon.exterior, *pred_polygon.interiors])
    gt_contours = shapely.geometry.GeometryCollection([gt_polygon.exterior, *gt_polygon.interiors])

    sampled_pred_contours = sample_geometry(pred_contours, sampling_spacing)
    # Project sampled contour points to ground truth contours
    projected_pred_contours = project_onto_geometry(sampled_pred_contours, gt_contours, pool=None)
    contour_measures = []
    for contour, proj_contour in zip(sampled_pred_contours.geoms, projected_pred_contours.geoms):
        coords = np.array(contour.coords[:])
        proj_coords = np.array(proj_contour.coords[:])
        edges = coords[1:] - coords[:-1]
        proj_edges = proj_coords[1:] - proj_coords[:-1]
        # Remove edges with a norm of zero
        edge_norms = np.linalg.norm(edges, axis=1)
        proj_edge_norms = np.linalg.norm(proj_edges, axis=1)
        norm_valid_mask = 0 < edge_norms * proj_edge_norms
        edges = edges[norm_valid_mask]
        proj_edges = proj_edges[norm_valid_mask]
        edge_norms = edge_norms[norm_valid_mask]
        proj_edge_norms = proj_edge_norms[norm_valid_mask]
        # Remove edge that have stretched more than max_stretch (invalid projection)
        stretch = edge_norms / proj_edge_norms
        stretch_valid_mask = np.logical_and(1 / max_stretch < stretch, stretch < max_stretch)
        edges = edges[stretch_valid_mask]
        if edges.shape[0] == 0:
            # Invalid projection for the whole contour, skip it
            continue
        proj_edges = proj_edges[stretch_valid_mask]
        edge_norms = edge_norms[stretch_valid_mask]
        proj_edge_norms = proj_edge_norms[stretch_valid_mask]
        scalar_products = np.abs(np.sum(np.multiply(edges, proj_edges), axis=1) / (edge_norms * proj_edge_norms))
        try:
            contour_measures.append(scalar_products.min())
        except ValueError:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [contour])
            plot_geometries(ax[1], [proj_contour])
            plot_geometries(ax[2], gt_contours)
            fig.tight_layout()
            plt.show()

    if len(contour_measures):
        min_scalar_product = min(contour_measures)
        measure = np.arccos(min_scalar_product)
        measure = measure * 180 / np.pi
 
        return measure
    else:
        return None

def get_within_bounds_ids(bbox, bounds, type='has_intersection'):
    """
    type could be has_intersection or contain
    """
    if len(bounds) == 0:
        return bounds

    start_x, start_y, end_x, end_y = bbox

    # start_x, start_y, end_x, end_y = crop_box
    if type == 'contain':
        flag1 = bounds[:,0] >= start_x
        flag2 = bounds[:,2] < end_x
        flag3 = bounds[:,1] >= start_y
        flag4 = bounds[:,3] < end_y

        return flag1 & flag2 & flag3 & flag4
    elif type == 'has_intersection':
        flag1 = bounds[:,0] > end_x
        flag2 = bounds[:,2] < start_x
        flag3 = bounds[:,1] > end_y
        flag4 = bounds[:,3] < start_y

        return (~(flag1 | flag2)) & (~(flag3 | flag4))
    else:
        raise ValueError()

def compute_polygon_contour_measures(pred_polygons: list, gt_polygons: list, sampling_spacing:
                                     float, min_precision: float, max_stretch: float, metric_name:
                                     str="cosine", progressbar=False):
    """
    pred_polygons are sampled with sampling_spacing before projecting those sampled points to gt_polygons.
    Then the

    @param pred_polygons:
    @param gt_polygons:
    @param sampling_spacing:
    @param min_precision: Polygons in pred_polygons must have a precision with gt_polygons above min_precision to be included in further computations
    @param max_stretch:  Exclude edges that have been stretched by the projection more than max_stretch from further computation
    @param metric_name: Metric type, can be "cosine" or ...
    @return:
    """
    assert isinstance(pred_polygons, list), "pred_polygons should be a list"
    assert isinstance(gt_polygons, list), "gt_polygons should be a list"
    if len(pred_polygons) == 0 or len(gt_polygons) == 0:
        return [None]
    assert isinstance(pred_polygons[0], shapely.geometry.Polygon), \
        f"Items of pred_polygons should be of type shapely.geometry.Polygon, not {type(pred_polygons[0])}"
    assert isinstance(gt_polygons[0], shapely.geometry.Polygon), \
        f"Items of gt_polygons should be of type shapely.geometry.Polygon, not {type(gt_polygons[0])}"



    """
    Old method to calcuate matched polygons
    """
    """
    gt_polygons_ = shapely.geometry.collection.GeometryCollection(gt_polygons)
    pred_polygons_ = shapely.geometry.collection.GeometryCollection(pred_polygons)

    # Filter pred_polygons to have at least a precision with gt_polygons of min_precision
    filtered_pred_polygons_ = [pred_polygon for pred_polygon in pred_polygons_.geoms if min_precision < pred_polygon.intersection(gt_polygons_).area / pred_polygon.area]
    # Extract contours of gt polygons
    gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons_.geoms for contour in [polygon.exterior, *polygon.interiors]])
    # Measure metric for each pred polygon
    if progressbar:
        process_id = int(multiprocess.current_process().name[-1])
        iterator = tqdm(filtered_pred_polygons_, desc="Contour measure", leave=False, position=process_id)
    else:
        iterator = tqdm(filtered_pred_polygons_, desc='Computing MTAs...')
    half_tangent_max_angles_ = [compute_contour_measure(pred_polygon, gt_contours, sampling_spacing=sampling_spacing, max_stretch=max_stretch, metric_name=metric_name)
                               for pred_polygon in iterator]
    """

    # pred_bounds = np.array([polygon.bounds for polygon in pred_polygons])
    gt_bounds = np.array([polygon.bounds for polygon in gt_polygons])

    filtered_polygons = []
    for pred_polygon in pred_polygons:
        valid_inds = get_within_bounds_ids(pred_polygon.bounds, gt_bounds)
        valid_gt_polygons = [gt_polygons[x] for x in valid_inds.nonzero()[0]]
        for gt_polygon in valid_gt_polygons:
            if min_precision < pred_polygon.intersection(gt_polygon).area / (pred_polygon.area + 1e-8):
                filtered_polygons.append([pred_polygon, gt_polygon])
                break


    # gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons.geoms for contour in [polygon.exterior, *polygon.interiors]])
    gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons for contour in [polygon.exterior, *polygon.interiors]])
    half_tangent_max_angles = [compute_contour_measure(pred_polygon, gt_polygon, sampling_spacing=sampling_spacing, max_stretch=max_stretch, metric_name=metric_name)
                               for pred_polygon, gt_polygon in filtered_polygons]
                               # for pred_polygon, gt_polygon in tqdm(filtered_polygons, desc='Computing MTAs')]
    return half_tangent_max_angles


def fix_polygons(polygons, buffer=0.0):
    polygons_geom = shapely.ops.unary_union(polygons)  # Fix overlapping polygons
    polygons_geom = polygons_geom.buffer(buffer)  # Fix self-intersecting polygons and other things
    fixed_polygons = []
    if polygons_geom.geom_type == "MultiPolygon":
        for poly in polygons_geom.geoms:
            fixed_polygons.append(poly)
    elif polygons_geom.geom_type == "Polygon":
        fixed_polygons.append(polygons_geom)
    else:
        raise TypeError(f"Geom type {polygons_geom.geom_type} not recognized.")
    return fixed_polygons


POINTS = []

#
# def compute_half_tangent_measure(pred_polygon, gt_contours, step=0.1, metric_name="angle"):
#     """
#     For each vertex in pred_polygon, find the closest gt contour and the closest point on that contour. From that point, compute both half-tangents.
#     measure angle difference between half-tangents of pred and corresponding gt points.
#     @param pred_polygon:
#     @param gt_contours:
#     @param metric_name:
#     @return:
#     """
#     assert isinstance(pred_polygon, shapely.geometry.Polygon), "pred_polygon should be a shapely Polygon"
#     pred_contours = [pred_polygon.exterior, *pred_polygon.interiors]
#     tangent_measures_list = []
#     for pred_contour in pred_contours:
#         pos_array = np.array(pred_contour.coords[:])
#         pred_tangents = pos_array[1:] - pos_array[:-1]
#         gt_tangent_1_list = []
#         gt_tangent_2_list = []
#         for i, pos in enumerate(pos_array[:-1]):
#             pred_point = shapely.geometry.Point(pos)
#             dist_to_gt = np.inf
#             closest_gt_contour = None
#             for gt_contour in gt_contours:
#                 d = pred_point.distance(gt_contour)
#                 if d < dist_to_gt:
#                     dist_to_gt = d
#                     closest_gt_contour = gt_contour
#             gt_point_t = closest_gt_contour.project(pred_point)  # References the projection of pred_point onto closest_gt_contour with a 1d referencing coordinate t
#             # --- Compute tangents of projected point on gt:
#             gt_point_tangent_1 = closest_gt_contour.interpolate(gt_point_t - step)
#             POINTS.append(gt_point_tangent_1)
#             gt_point = closest_gt_contour.interpolate(gt_point_t)
#             POINTS.append(gt_point)
#             gt_point_tangent_2 = closest_gt_contour.interpolate(gt_point_t + step)
#             POINTS.append(gt_point_tangent_2)
#             gt_pos_tangent_1 = np.array(gt_point_tangent_1.coords[0])
#             gt_pos_tangent_2 = np.array(gt_point_tangent_2.coords[0])
#             gt_pos = np.array(gt_point.coords[0])
#             gt_tangent_1 = gt_pos_tangent_1 - gt_pos
#             gt_tangent_2 = gt_pos_tangent_2 - gt_pos
#             gt_tangent_1_list.append(gt_tangent_1)
#             gt_tangent_2_list.append(gt_tangent_2)
#         gt_tangents_1 = np.stack(gt_tangent_1_list, axis=0)
#         gt_tangents_2 = np.stack(gt_tangent_2_list, axis=0)
#         # Measure dist between pred_tangents and gt_tangents
#         pred_norms = np.linalg.norm(pred_tangents, axis=1)
#         tangent_1_measures = np.abs(np.sum(np.multiply(np.roll(pred_tangents, 1, axis=0), gt_tangents_1), axis=1) / (np.roll(pred_norms, 1, axis=0) * step))
#         tangent_2_measures = np.abs(np.sum(np.multiply(pred_tangents, gt_tangents_2), axis=1) / (pred_norms * step))
#         print(tangent_1_measures)
#         print(tangent_2_measures)
#         tangent_measures_list.append(tangent_1_measures)
#         tangent_measures_list.append(tangent_2_measures)
#     tangent_measures = np.concatenate(tangent_measures_list)
#     min_scalar_product = np.min(tangent_measures)
#     max_angle = np.arccos(min_scalar_product)
#     return max_angle

#
# def compute_vertex_measures(pred_polygons: list, gt_polygons: list, min_precision: float, metric_name: str="angle", pool: Pool=None):
#     """
#     Computes measure for each pred_polygon
#     @param pred_polygons:
#     @param gt_polygons:
#     @param min_precision:
#     @param metric_name:
#     @param pool:
#     @return:
#     """
#     assert isinstance(pred_polygons, list), "pred_polygons should be a list"
#     assert isinstance(gt_polygons, list), "gt_polygons should be a list"
#     if len(pred_polygons) == 0 or len(gt_polygons) == 0:
#         return np.array([]), [], []
#     assert isinstance(pred_polygons[0], shapely.geometry.Polygon), \
#         f"Items of pred_polygons should be of type shapely.geometry.Polygon, not {type(pred_polygons[0])}"
#     assert isinstance(gt_polygons[0], shapely.geometry.Polygon), \
#         f"Items of gt_polygons should be of type shapely.geometry.Polygon, not {type(gt_polygons[0])}"
#     gt_polygons = shapely.geometry.collection.GeometryCollection(gt_polygons)
#     pred_polygons = shapely.geometry.collection.GeometryCollection(pred_polygons)
#     # Filter pred_polygons to have at least a precision with gt_polygons of min_precision
#     filtered_pred_polygons = [pred_polygon for pred_polygon in pred_polygons if min_precision < pred_polygon.intersection(gt_polygons).area / pred_polygon.area]
#     # Extract contours of gt polygons
#     gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons for contour in [polygon.exterior, *polygon.interiors]])
#     # Measure metric for each pre polygon
#     half_tangent_max_angles = [compute_half_tangent_measure(pred_polygon, gt_contours, metric_name=metric_name)
#                                for pred_polygon in filtered_pred_polygons]
#     return half_tangent_max_angles

def map_nearest_nonzero(A, B):
    B_nonzero = B != 0

    # Compute the inverted binary mask where zeros are True and non-zeros are False
    inv_mask = np.logical_not(B_nonzero)

    # Use distance transform to find the nearest nonzero pixel indices
    distances, indices = distance_transform_edt(inv_mask, return_indices=True)

    # Extract rows and columns indices of nearest non-zero elements
    nearest_r = indices[0]
    nearest_c = indices[1]

    # Use these indices to map colors from B to A
    C = np.zeros_like(B)
    A_mask = A == 1
    C[A_mask] = B[nearest_r[A_mask], nearest_c[A_mask]]

    return C

def cal_iou(polygon1, polygon2, eps=1e-8):
    intersection = polygon1.intersection(polygon2)
    union = polygon1.union(polygon2)
    iou = intersection.area / (union.area + eps)
    return iou

def sample_points_in_ring(linear_ring, interval, num_min_bins=8):
    """
    Interpolates points on a ring with unequal segment lengths using parameters ts.

    :param points: NumPy array of shape (N, 2) representing N 2-D points on the ring.
    :param ts: NumPy array of parameters for interpolation, where each element is in [0, 1].
    :return: NumPy array of shape (len(ts), 2) representing the interpolated points on the ring.
    """

    points = np.array(linear_ring.coords)[:-1]

    N = points.shape[0]  # Number of points

    # Calculate segment lengths
    segment_lengths = np.sqrt(((points - np.roll(points, -1, axis=0))**2).sum(axis=1))
    perimeter = segment_lengths.sum()

    num_bins = max(round(perimeter / interval), num_min_bins)
    # num_bins = min(num_bins, N)
    ts = np.linspace(0, 1, num_bins)
    ts = np.mod(ts, 1)

    # Calculate cumulative length proportions
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths))) / perimeter
    
    # Function to find the segment index for each t
    def find_segment_index(t, cumulative_lengths):
        return np.searchsorted(cumulative_lengths, t, side='right') - 1

    # Map ts to segment indices
    segment_indices = find_segment_index(ts, cumulative_lengths)

    # Calculate t' for each segment
    t_primes = (ts - cumulative_lengths[segment_indices]) / (segment_lengths[segment_indices] / perimeter)
    
    # Interpolate within the selected segments
    start_points = points[segment_indices]
    end_points = points[(segment_indices + 1) % N]  # Wrap around to the first point for the last segment
    interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]

    return interpolated_points

def sample_points_in_ring_numpy(ring, interval=4, num_min_bins=8):
    """
    Interpolates points on a ring with unequal segment lengths using parameters ts.

    :param points: NumPy array of shape (N, 2) representing N 2-D points on the ring.
    :param ts: NumPy array of parameters for interpolation, where each element is in [0, 1].
    :return: NumPy array of shape (len(ts), 2) representing the interpolated points on the ring.
    """

    points = ring[:-1]

    N = points.shape[0]  # Number of points

    # Calculate segment lengths
    segment_lengths = np.sqrt(((points - np.roll(points, -1, axis=0))**2).sum(axis=1))
    perimeter = segment_lengths.sum()

    num_bins = max(round(perimeter / interval), num_min_bins)
    # num_bins = min(num_bins, N)
    ts = np.linspace(0, 1, num_bins)
    ts = np.mod(ts, 1)

    # Calculate cumulative length proportions
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths))) / perimeter
    
    # Function to find the segment index for each t
    def find_segment_index(t, cumulative_lengths):
        return np.searchsorted(cumulative_lengths, t, side='right') - 1

    # Map ts to segment indices
    segment_indices = find_segment_index(ts, cumulative_lengths)

    # Calculate t' for each segment
    t_primes = (ts - cumulative_lengths[segment_indices]) / (segment_lengths[segment_indices] / perimeter)
    
    # Interpolate within the selected segments
    start_points = points[segment_indices]
    end_points = points[(segment_indices + 1) % N]  # Wrap around to the first point for the last segment
    interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]

    return interpolated_points

def sample_rings(polygons, interval=2, length=50, ring_stride=25):
    """
    polygons: json format
    """

    def sample_rings_fun(polygons, interval, length, ring_stride):

        def sample_points_in_ring(ring, interval=2):

            # sampled_points = self.interpolate_ring_unequal_lengths(np.array(ring[:-1]), interval)

            N = ring.shape[0]  # Number of points

            # Calculate segment lengths
            segment_lengths = np.sqrt(((ring - np.roll(ring, -1, axis=0))**2).sum(axis=1))
            perimeter = segment_lengths.sum()

            num_bins = max(round(perimeter / interval), 8)
            num_bins = max(num_bins, N)
            ts = np.linspace(0, 1, num_bins)
            ts = np.mod(ts, 1)

            # Calculate cumulative length proportions
            cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths))) / perimeter

            # Function to find the segment index for each t
            def find_segment_index(t, cumulative_lengths):
                return np.searchsorted(cumulative_lengths, t, side='right') - 1

            # Map ts to segment indices
            segment_indices = find_segment_index(ts, cumulative_lengths)

            # Calculate t' for each segment
            t_primes = (ts - cumulative_lengths[segment_indices]) / (segment_lengths[segment_indices] / perimeter)

            # Interpolate within the selected segments
            start_points = ring[segment_indices]
            end_points = ring[(segment_indices + 1) % N]  # Wrap around to the first point for the last segment
            interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]

            return interpolated_points

        def separate_ring(ring, crop_len, stride):

            N, _ = ring.shape
            if N < crop_len:
                ring = np.concatenate([ring, np.zeros((crop_len-N, 2)) - 1])
                return [ring]

            repeated_ring = np.concatenate([ring[:-1], ring], axis=0)

            num_parts = math.ceil((N - crop_len) / stride) \
                    if math.ceil((N - crop_len) / stride) * stride + crop_len >= N \
                    else math.ceil((N - crop_len) / stride) + 1

            idxes = np.arange(num_parts + 1)  * stride
            # offset = np.where(idxes + crop_len > N, N - crop_len, idxes)

            rings = [repeated_ring[x:x + crop_len] for x in idxes]
            return rings

        all_rings = []
        all_ring_sizes = []
        all_idxes = []

        # for i, polygon in tqdm(enumerate(polygons), desc='sampling rings...'):
        for i, polygon in enumerate(polygons):
            # rings = np.array(polygon['coordinates'])
            rings = polygon['coordinates']
            sizes = []
            if len(rings) == 0:
                pdb.set_trace()
            for j, ring in enumerate(rings):
                sampled_ring = np.array(sample_points_in_ring(np.array(ring[:-1]), interval=interval))
                ring_parts = np.array(separate_ring(sampled_ring, crop_len=length, stride=ring_stride))
                idx1 = np.array([i] * len(ring_parts))
                idx2 = np.array([j] * len(ring_parts))
                idx = np.stack([idx1, idx2], axis=1)
                all_rings.append(ring_parts)
                all_idxes.append(idx)
                sizes.append(len(sampled_ring))

            all_ring_sizes.append(sizes)

        return all_rings, all_ring_sizes, all_idxes

    all_rings, all_ring_sizes, all_idxes = sample_rings_fun(polygons, interval, length, ring_stride)

    if len(all_rings) == 0:
        return torch.zeros(0,2), torch.zeros(0,), torch.zeros(0,)

    all_rings = np.concatenate(all_rings, axis=0)
    all_idxes = np.concatenate(all_idxes, axis=0)

    return all_rings, all_idxes, all_ring_sizes


def calculate_angle_between_points(points):
    # Calculate the differences between consecutive points
    diffs = points[1:] - points[:-1]

    # Calculate the angles
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    # Convert angles to degrees
    angles_degrees = np.degrees(angles)

    return angles_degrees

def calculate_angles_between_segments(points):
    # Calculate differences between consecutive points
    diffs = points[1:] - points[:-1]
    
    # Calculate the dot product of consecutive segments
    dot_products = np.sum(diffs[:-1] * diffs[1:], axis=1)
    
    # Calculate the magnitudes of the segments
    magnitudes = np.linalg.norm(diffs[:-1], axis=1) * np.linalg.norm(diffs[1:], axis=1)
    
    # Find indices where magnitudes are not zero
    nonzero_indices = magnitudes != 0
    
    # Calculate the cosine of the angles between segments, handling zero magnitudes
    cos_angles = np.zeros_like(dot_products)
    cos_angles[nonzero_indices] = dot_products[nonzero_indices] / magnitudes[nonzero_indices]
    
    # Clip values to ensure they are within the valid range for arccosine
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    
    # Calculate the angles in radians
    angles_radians = np.arccos(cos_angles)
    
    # Convert angles to degrees
    # angles_degrees = np.degrees(angles_radians)
    
    return angles_radians

def calculate_dot_products(edges, proj_edges, max_stretch=2.0):

    edge_norms = torch.norm(edges, dim=-1)
    proj_edge_norms = torch.norm(proj_edges, dim=-1)
    norm_valid_mask = 0 < edge_norms * proj_edge_norms
    edges = edges[norm_valid_mask]
    proj_edges = proj_edges[norm_valid_mask]
    edge_norms = edge_norms[norm_valid_mask]
    proj_edge_norms = proj_edge_norms[norm_valid_mask]
    # Remove edge that have stretched more than max_stretch (invalid projection)
    stretch = edge_norms / proj_edge_norms
    stretch_valid_mask = (1 / max_stretch < stretch) & (stretch < max_stretch)
    edges = edges[stretch_valid_mask]
    if edges.shape[0] == 0:
        # Invalid projection for the whole contour, skip it
        return None

    proj_edges = proj_edges[stretch_valid_mask]
    # edge_norms = edge_norms[stretch_valid_mask]
    # proj_edge_norms = proj_edge_norms[stretch_valid_mask]
    # scalar_products = torch.abs((edges * proj_edges) / (edge_norms * proj_edge_norms).sum())
    scalar_products = F.cosine_similarity(edges, proj_edges)
    scalar_products = torch.clip(scalar_products, -1.0, 1.0)
    measure = torch.arccos(scalar_products)
    if torch.isnan(measure.sum()):
        pdb.set_trace()

    return measure


def sample_rings_by_features(batch_gt_polygons, ring_len=20, pixel_width=0.3, scale_factor=4, input_poly_type='noise'):

    def sample_points_in_ring(ring, interval=None):

        try:
            ring_shape = shapely.LinearRing(ring)
        except ValueError:
            return None

        perimeter = ring_shape.length
        num_bins = max(round(perimeter / interval), 4)
        num_bins = max(num_bins, len(ring))

        bins = np.linspace(0, 1, num_bins)
        sampled_points = [ring_shape.interpolate(x, normalized=True) for x in bins]
        sampled_points = [[temp.x, temp.y] for temp in sampled_points]

        return sampled_points

    def add_noise_to_ring(ring, interval=4, noise_type='uniform'):

        if noise_type == 'random':
            noise_type = random.choice(['uniform', 'skip'])

        if noise_type == 'uniform':
            noise = (np.random.rand(len(ring), 2) - 0.5) * interval / 2.
        elif noise_type == 'skip':
            noise = (np.random.rand(len(ring), 2) - 0.5) * interval
            noise[0:2:-1] = 0

        noisy_ring = ring + noise

        return noisy_ring

    def get_target_ring(ring_A, ring_B):
        sampled_points = self.sample_points_in_ring(ring_A)
        ring_A = torch.tensor(np.array(sampled_points))[:-1]
        ring_B = torch.tensor(np.array(ring_B))[:-1]

        if len(ring_A) < len(ring_B):
            return None, None, None

        assign_result = self.assigner.assign(ring_A, ring_B)

        ring_A_cls_target = torch.zeros(len(ring_A), dtype=torch.long)
        ring_A_reg_target = torch.zeros(len(ring_A), 2)
        segments_A = torch.zeros(len(ring_A), 4)

        temp = assign_result.gt_inds - 1
        temp2 = (temp > -1).nonzero().view(-1)
        ring_A_cls_target[temp2] = 1
        ring_A_reg_target[temp2] = ring_B[temp[temp2]].float()

        ring_A_cls_target.nonzero().view(-1)
        segments_A[:temp2[0]] = torch.cat(
                [ring_A_reg_target[temp2[-1]], ring_A_reg_target[temp2[0]]]
        ).view(1, -1)
        for i, idx in enumerate(temp2):
            left = idx
            right = temp2[i+1] if i < len(temp2) - 1 else len(ring_A)

            seg_x = idx
            seg_y = temp2[i+1] if i < len(temp2) - 1 else temp2[0]

            segments_A[left:right] = torch.cat(
                [ring_A_reg_target[seg_x], ring_A_reg_target[seg_y]]
            ).view(1, -1)

        projs = self.project_points_onto_segments(segments_A[ring_A_cls_target==0], ring_A[ring_A_cls_target==0])
        ring_A_reg_target[ring_A_cls_target==0] = torch.tensor(projs).float()

        return ring_A, ring_A_cls_target, ring_A_reg_target


    batch_rings = []
    batch_ring_cls_targets = []
    batch_ring_reg_targets = []
    batch_ring_angle_targets = []

    for i, gt_polygons in enumerate(batch_gt_polygons):
        rings = []
        ring_cls_targets = []
        ring_reg_targets = []
        ring_angle_targets = []

        for polygon in gt_polygons:
            buffer = 2
            exterior = np.array(polygon['coordinates'][0])
            norm_exterior = (exterior - exterior.min(axis=0, keepdims=True)) / pixel_width + buffer

            """
            sample rings
            """
            shape = (norm_exterior.max(axis=0) + buffer).round().astype(np.int)
            shape = shape[[1,0]]
            noisy_norm_exterior = add_noise_to_ring(norm_exterior)
            raster = rasterio.features.rasterize([shapely.Polygon(noisy_norm_exterior)], out_shape=shape, dtype=np.int32, all_touched=False)

            if raster.sum() == 0:
                continue

            polygonized = next(rasterio.features.shapes(raster, mask=raster > 0))
            ring = polygonized[0]['coordinates'][0]
            ring, cls_target, reg_target = get_target_ring(ring, norm_exterior)

            if ring is None:
                continue

            ring = ring.round().to(torch.int)


            """
            calculate targets
            """

            start_idx = random.randint(0, len(ring)-1)
            shuffled_ring = torch.cat([ring[start_idx:], ring[:start_idx]])
            shuffled_cls_target = torch.cat([cls_target[start_idx:], cls_target[:start_idx]])
            shuffled_reg_target = torch.cat([reg_target[start_idx:], reg_target[:start_idx]])

            ext_polygon = torch.cat([shuffled_reg_target, shuffled_reg_target[0:1]])
            diff = ext_polygon[1:] - ext_polygon[:-1]
            degs = torch.fmod(torch.arctan2(- diff[:,1], diff[:,0]), 2 * np.pi)
            # bins = torch.linspace(0, 2 * math.pi, num_bins + 1)
            # bin_indices = torch.searchsorted(bins, degs, right=True)  # right closed
            # shuffled_angle_target = torch.eye(num_bins)[bin_indices - 1]
            shuffled_angle_target = degs / np.pi

            sampled_ring = torch.zeros(ring_len, 2, dtype=shuffled_ring.dtype) - 1
            sampled_cls_target = torch.zeros(ring_len, dtype=shuffled_cls_target.dtype)
            sampled_reg_target = torch.zeros(ring_len, 2, dtype=shuffled_reg_target.dtype) - 1
            sampled_angle_target = torch.zeros(ring_len, dtype=shuffled_reg_target.dtype)

            sampled_ring[:len(shuffled_ring)] = shuffled_ring[:ring_len]
            sampled_cls_target[:len(shuffled_ring)] = shuffled_cls_target[:ring_len]
            sampled_reg_target[:len(shuffled_ring)] = shuffled_reg_target[:ring_len]
            sampled_angle_target[:len(shuffled_ring)] = shuffled_angle_target[:ring_len]

            if sampled_cls_target.sum() >= 1:
                rings.append(sampled_ring)
                ring_cls_targets.append(sampled_cls_target)
                ring_reg_targets.append(sampled_reg_target)
                ring_angle_targets.append(sampled_angle_target)


        if len(rings) > 0:
            num_max_ring = self.ring_sample_conf.get('num_max_ring', 512)
            batch_rings.append(torch.stack(rings)[:num_max_ring])
            batch_ring_cls_targets.append(torch.stack(ring_cls_targets)[:num_max_ring])
            batch_ring_reg_targets.append(torch.stack(ring_reg_targets)[:num_max_ring])
            batch_ring_angle_targets.append(torch.stack(ring_angle_targets)[:num_max_ring])
        else:
            batch_rings.append(torch.ones(1, ring_len, 2)) # dummy points
            batch_ring_cls_targets.append(torch.zeros(1, ring_len, dtype=torch.long))
            batch_ring_reg_targets.append(torch.ones(1, ring_len, 2))
            batch_ring_angle_targets.append(torch.zeros(1, ring_len))

    return batch_rings, batch_ring_cls_targets, batch_ring_reg_targets, batch_ring_angle_targets

def cluster_points_by_sim(A, eps=0.5, min_samples=5):

    # cluster_fun = AffinityPropagation()
    # cluster_idx = cluster_fun.fit_predict(A)

    # density = np.array([(A[i] >= eps).sum().item() for i in range(len(A))])
    # idx = np.argsort(density)[::-1]
    # pdb.set_trace()

    dis = torch.where(1 - A > 0, 1 - A, 0).detach().cpu().numpy()
    cluster_fun = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
    cluster_idx = cluster_fun.fit_predict(dis)

    return cluster_idx

    last = -2
    idxes = []
    for i, idx in enumerate(cluster_idx):
        idx = idx.item()
        if (idx != last) or (idx == -1):
            idxes.append(i)
            last = idx

    return idxes


def cluster_to_polygons(ring, angle, cluster_idx):

    def line_intersection(m1, b1, m2, b2):
        """
        Calculate the intersection point of two lines given their slopes and intercepts.
        """
        if abs(m1 - m2) < 1e-7:
            return None
        x_intersection = (b2 - b1) / (m1 - m2)
        y_intersection = m1 * x_intersection + b1
        return x_intersection, y_intersection

    N = cluster_idx.max() + 1
    num_minus = (cluster_idx == -1).sum()
    cluster_idx[cluster_idx == -1] = np.arange(N, N+num_minus)
    N = cluster_idx.max() + 1

    centers = []
    angles = []
    end_points = []
    ms = []
    bs = []
    unique, index = np.unique(cluster_idx, return_index=True)
    sorted_cls_idx = unique[index.argsort()]

    for i in sorted_cls_idx:
        X = ring[cluster_idx == i]
        # model = LinearRegression()
        # model.fit(X[:,0:1], X[:,1])
        # ms.append(model.coef_[0])
        # bs.append(model.intercept_)

        center = ring[cluster_idx == i].mean(dim=0)
        mean_angle = angle[cluster_idx == i].mean()

        centers.append(center)
        angles.append(mean_angle)
        end_points.append(X[0])

    centers = torch.stack(centers)
    angles = torch.stack(angles)
    end_points = torch.stack(end_points)

    return end_points

    # return end_points

    # ms = np.array(ms + ms[0:1])
    # bs = np.array(bs + bs[0:1])

    # points = []
    # for i in range(len(ms) - 1):
    #     point = line_intersection(ms[i], bs[i], ms[i+1], bs[i+1])
    #     if point is not None:
    #         points.append(point)
    #     else:
    #         points.append(ring[cluster_idx == i][-1].tolist())

    P1 = centers
    P2 = torch.roll(centers, shifts=-1, dims=0)
    R1 = angles
    R2 = torch.roll(angles, shifts=-1, dims=0)

    pdb.set_trace()
    # points = np.stack(calculate_intersecting_points(P1, P2, R1, R2))
    # model = LinearRegression()
    # model.fit(x, y)

    return torch.tensor(points)


def transform_polygon(polygon, affine_matrix):
    import numpy as np
    from shapely.geometry import Polygon
    """
    Transform the coordinates of a Polygon object and its interior rings using an Affine matrix.
    
    Args:
    - polygon: Shapely Polygon object with pixel coordinates
    - affine_matrix: Affine matrix (list or numpy array) for coordinate transformation
    
    Returns:
    - transformed_polygon: Transformed Polygon object with coordinates transformed by the Affine matrix
    """
    # Convert the Polygon coordinates to homogeneous coordinates

    affine_matrix = np.array(affine_matrix)[:6].reshape(2,3)

    exterior_coords = np.concatenate([np.array(polygon.exterior.coords), np.ones((len(np.array(polygon.exterior.coords)), 1))], axis=1)
    # exterior_coords = np.column_stack((np.array(polygon.exterior.xy), np.ones(len(polygon.exterior.xy[0]))))
    transformed_exterior_coords = np.dot(affine_matrix, exterior_coords.T).T
    
    # Convert the homogeneous coordinates back to Cartesian coordinates
    transformed_exterior_coords_cartesian = transformed_exterior_coords[:, :2] / transformed_exterior_coords[:, 2][:, None]
    
    # Create a new Polygon object with the transformed exterior coordinates
    transformed_exterior = Polygon(transformed_exterior_coords_cartesian)
    
    # Transform the coordinates of the interior rings, if any
    transformed_interiors = []
    for interior in polygon.interiors:
        interior_coords = np.column_stack((np.array(interior.xy), np.ones(len(interior.xy[0]))))
        transformed_interior_coords = np.dot(affine_matrix, interior_coords.T).T
        transformed_interior_coords_cartesian = transformed_interior_coords[:, :2] / transformed_interior_coords[:, 2][:, None]
        transformed_interiors.append(transformed_interior_coords_cartesian)
    
    # Create Polygon objects for the transformed interior rings
    transformed_interiors_polygons = [Polygon(interior) for interior in transformed_interiors]
    
    # Create the transformed Polygon object with exterior and interior rings
    transformed_polygon = Polygon(transformed_exterior.exterior, transformed_interiors_polygons)
    
    return transformed_polygon


from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
from tqdm import tqdm

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou

def compute_IoU_cIoU(coco, coco_gti, score_thre=0.5):

    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        N = 0
        is_first = True
        for _idx, annotation in enumerate(annotations):
            if annotation['score'] > score_thre:
                rle = cocomask.frPyObjects(annotation['polygon'], img['height'], img['width'])
                m = cocomask.decode(rle)
                # m = np.mod(m.sum(axis=-1), 2)
                m = m.sum(axis=-1)

                if is_first:
                    mask = m.reshape((img['height'], img['width']))
                    N = len(annotation['polygon'][0]) // 2
                    is_first = False
                else:
                    mask = mask + m.reshape((img['height'], img['width']))
                    N = N + len(annotation['polygon'][0]) // 2

        mask = mask != 0


        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        N_GT = 0
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            if _idx == 0:
                mask_gti = m.reshape((img['height'], img['width']))
                N_GT = len(annotation['segmentation'][0]) // 2
            else:
                mask_gti = mask_gti + m.reshape((img['height'], img['width']))
                N_GT = N_GT + len(annotation['segmentation'][0]) // 2

        mask_gti = mask_gti != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)

        bar.set_description("iou: %2.4f, c-iou: %2.4f" % (np.mean(list_iou), np.mean(list_ciou)))
        bar.refresh()

    # print("Done!")
    # print("Mean IoU: ", np.mean(list_iou))
    # print("Mean C-IoU: ", np.mean(list_ciou))
    return list_iou, list_ciou

def compute_mta(coco_eval, score_thre=0.5):
    mtas = []
    for img_id in coco_eval.params.imgIds:
        gts = coco_eval.cocoGt.loadAnns(coco_eval.cocoGt.getAnnIds(imgIds=[img_id]))
        dts = coco_eval.cocoDt.loadAnns(coco_eval.cocoDt.getAnnIds(imgIds=[img_id]))
        if len(gts) > 0 and len(dts) > 0:
            gt_polygons = []
            for gt in gts:
                ext = np.array(gt['segmentation'][0]).reshape(-1,2).tolist()
                gt_polygon = shapely.Polygon(shell=ext)
                if gt_polygon.is_valid:
                    gt_polygons.append(gt_polygon)

            dt_polygons = []
            for dt in dts:
                if dt['score'] > score_thre:
                    ext = np.array(dt['polygon'][0]).reshape(-1,2).tolist()
                    cur_polygon = shapely.Polygon(shell=ext)
                    fixed_polygon = cur_polygon.buffer(0.0)  # Fix self-intersecting polygons and other things
                    dt_polygons.append(fixed_polygon)

            fixed_gt_polygons = fix_polygons(gt_polygons, buffer=0.0)
            # fixed_dt_polygons = fix_polygons(dt_polygons, buffer=0.0)
            fixed_dt_polygons = dt_polygons

            cur_mtas = compute_polygon_contour_measures(fixed_dt_polygons, fixed_gt_polygons, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)
            mtas.extend(cur_mtas)

    return mtas

def get_nearest_ring(ring1, ring2):
    projected_ring1 = [shapely.ops.nearest_points(shapely.geometry.Point(x), ring2)[1] for x in ring1.coords]
    projected_ring1 = np.concatenate([np.array(x.coords) for x in projected_ring1], axis=0)

    return projected_ring1

def fast_searchsorted(A, B):
    A_idx = 0
    idxes = torch.zeros(len(B), dtype=torch.long) + len(A)
    for i in range(len(B)):
        while A_idx < len(A) and B[i] >= A[A_idx]:
            A_idx += 1

        if A_idx >= len(A):
            break

        idxes[i] = A_idx

    idxes[-1] = idxes[0]
    # idxes2 = torch.searchsorted(A, B, side='right')
    # assert (idxes == idxes2).all()

    return idxes


def interpolate_ring(linear_ring, drop_last=True, interval=None, num_bins=None, pad_length=None,
                     num_min_bins=8, num_max_bins=512, type='torch'):
    """
    Interpolates points on a ring with unequal segment lengths using parameters ts.

    :param points: NumPy array of shape (N, 2) representing N 2-D points on the ring.
    :param ts: NumPy array of parameters for interpolation, where each element is in [0, 1].
    :return: NumPy array of shape (len(ts), 2) representing the interpolated points on the ring.
    """
    assert type == 'numpy' or type == 'torch'
    assert interval is not None or num_bins is not None

    points = linear_ring[:-1] if drop_last else linear_ring
    N = points.shape[0]  # Number of points

    if type == 'numpy':

        segment_lengths = np.sqrt(((points - np.roll(points, -1, axis=0))**2).sum(axis=1))
        perimeter = segment_lengths.sum()
        if num_bins is None:
            num_bins = max(round(perimeter / interval), num_min_bins)

        # num_bins = min(num_bins, N)
        ts = np.linspace(0, 1, num_bins)
        ts = np.mod(ts, 1)

        # Calculate cumulative length proportions
        cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths))) / perimeter

        # Function to find the segment index for each t
        def find_segment_index(t, cumulative_lengths):
            return np.searchsorted(cumulative_lengths, t, side='right') - 1

        # Map ts to segment indices
        segment_indices = find_segment_index(ts, cumulative_lengths)

        # Calculate t' for each segment
        t_primes = (ts - cumulative_lengths[segment_indices]) / (segment_lengths[segment_indices] / perimeter)
        
        # Interpolate within the selected segments
        start_points = points[segment_indices]
        end_points = points[(segment_indices + 1) % N]  # Wrap around to the first point for the last segment
        interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]

        return interpolated_points

    elif type == 'torch':

        segment_lengths = torch.sqrt(((points - torch.roll(points, shifts=[-1], dims=[0]))**2).sum(dim=1))
        perimeter = segment_lengths.sum().item()
        if num_bins is None:
            num_bins = max(round(perimeter / interval), num_min_bins)
            if pad_length is not None:
                num_bins = min(num_bins, pad_length)
            num_bins = min(num_bins, num_max_bins)

        ts = torch.linspace(0, 1, num_bins)
        ts = torch.fmod(ts, 1)

        cumulative_lengths = torch.cat([torch.zeros(1), torch.cumsum(segment_lengths, dim=0)]) / perimeter

        segment_indices = torch.searchsorted(cumulative_lengths, ts, side='right') - 1
        # segment_indices = fast_searchsorted(cumulative_lengths, ts) - 1

        t_primes = (ts - cumulative_lengths[segment_indices]) / (segment_lengths[segment_indices] / perimeter)

        start_points = points[segment_indices]
        end_points = points[(segment_indices + 1) % N]  # Wrap around to the first point for the last segment
        interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]
        # gt_inds = ((interpolated_points[:-1].unsqueeze(0) - points.unsqueeze(1)) ** 2).sum(dim=-1).argmin(dim=1)
        if pad_length is not None and num_bins < pad_length:
            interpolated_points = F.pad(interpolated_points, (0,0,0,pad_length-num_bins), value=-1)

        return interpolated_points

def interpolate_segments(segments, num_bins=4):
    """
    Interpolates points on a ring with unequal segment lengths using parameters ts.

    :param points: NumPy array of shape (N, 2) representing N 2-D points on the ring.
    :param ts: NumPy array of parameters for interpolation, where each element is in [0, 1].
    :return: NumPy array of shape (len(ts), 2) representing the interpolated points on the ring.
    """
    N = segments.shape[0]  # Number of points

    segment_lengths = torch.sqrt(((segments[1:] - segments[:-1]) ** 2).sum(dim=1))
    perimeter = segment_lengths.sum().item()

    ts = torch.linspace(0, 1, num_bins)[:-1]
    # ts = torch.fmod(ts, 1)[:-1]

    cumulative_lengths = torch.cat([torch.zeros(1), torch.cumsum(segment_lengths, dim=0)]) / perimeter

    segment_indices = torch.searchsorted(cumulative_lengths, ts, side='right') - 1
    # segment_indices = fast_searchsorted(cumulative_lengths, ts) - 1

    t_primes = (ts - cumulative_lengths[segment_indices]) / (segment_lengths[segment_indices] / perimeter)

    start_points = segments[segment_indices]
    end_points = segments[(segment_indices + 1)]  # Wrap around to the first point for the last segment
    interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]

    return torch.cat([interpolated_points, segments[-1:]], dim=0)


def align_rings(ring_A, ring_B, num_primitive_queries):
    # ring_B_inter = F.interpolate(
    #     ring_B.unsqueeze(0).permute(0,2,1), size=(len(ring_A) + 1), mode='linear', align_corners=True
    # ).permute(0,2,1)[0, :-1]
    valid_mask = ~(ring_A == -1).all(dim=-1)
    ring_A = ring_A[valid_mask]

    ring_B_inter = interpolate_ring(ring_B, num_bins=len(ring_A), pad_length=None)

    A = ring_A[:-1]
    B = ring_B_inter[:-1]

    B_rolls = []
    for offset in range(len(B) - 1):
        B_roll = torch.roll(B, shifts=[offset], dims=[0])
        B_rolls.append(B_roll)

    B_rolls = torch.stack(B_rolls, dim=0)

    dis = ((A.unsqueeze(0) - B_rolls) ** 2).sum(dim=-1) ** 0.5
    roll_idx = dis.sum(dim=1).argmin(dim=0)
    aligned_B = B_rolls[roll_idx.item()]
    aligned_B = torch.cat([aligned_B, aligned_B[0:1]])

    gt_inds = ((aligned_B[:-1].unsqueeze(0) - ring_B[:-1].unsqueeze(1)) ** 2).sum(dim=-1).argmin(dim=1)
    # offsets = ring_B[:-1] - ring_A[gt_inds]
    offsets = ring_B[:-1]

    idx = gt_inds.argmin()
    gt_inds = torch.roll(gt_inds, shifts=[-idx.item()])
    offsets = torch.roll(offsets, shifts=[-idx.item()], dims=[0])

    if len(gt_inds) > num_primitive_queries:
        inds = torch.linspace(0, len(gt_inds) - 1, num_primitive_queries).round().int()
        gt_inds = gt_inds[inds]
        offsets = offsets[inds]
    else:
        gt_inds = F.pad(gt_inds, pad=(0,num_primitive_queries-len(ring_B)+1), value=-1)
        offsets = F.pad(offsets, pad=(0,0,0,num_primitive_queries-len(ring_B)+1), value=-1)

    return aligned_B, gt_inds, offsets

def align_rings_v2(ring_A, ring_B, num_primitive_queries):

    ring_B_inter = interpolate_ring(ring_B, num_bins=len(ring_A), pad_length=None)
    points = ring_B_inter[:-1]

    center = points.mean(dim=0)

    # Step 2: Compute the angles of each point relative to the center
    # Vector from center to each point
    vectors = points - center

    # Compute the angle of each vector
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])

    # Step 3: Find the point with the angle closest to 0
    # Wrap angles to the range [-pi, pi]
    angles = (angles + 2 * torch.pi) % (2 * torch.pi) - torch.pi

    # Find the index of the minimum angle (closest to 0)
    closest_to_x_axis_idx = torch.argmin(torch.abs(angles))

    aligned_B = torch.roll(ring_B_inter[:-1], shifts=[-closest_to_x_axis_idx], dims=[0])
    aligned_B = torch.cat([aligned_B, aligned_B[0:1]])

    gt_inds = ((aligned_B[:-1].unsqueeze(0) - ring_B[:-1].unsqueeze(1)) ** 2).sum(dim=-1).argmin(dim=1)
    # offsets = ring_B[:-1] - ring_A[gt_inds]
    offsets = ring_B[:-1]
    idx = gt_inds.argmin()
    gt_inds = torch.roll(gt_inds, shifts=[-idx.item()])
    offsets = torch.roll(offsets, shifts=[-idx.item()], dims=[0])

    if len(gt_inds) > num_primitive_queries:
        inds = torch.linspace(0, len(gt_inds) - 1, num_primitive_queries).round().int()
        gt_inds = gt_inds[inds]
        offsets = offsets[inds]
    else:
        gt_inds = F.pad(gt_inds, pad=(0,num_primitive_queries-len(ring_B)+1), value=-1)
        offsets = F.pad(offsets, pad=(0,0,0,num_primitive_queries-len(ring_B)+1), value=-1)

    return aligned_B, gt_inds, offsets

def align_rings_v2(ring_A, ring_B, num_primitive_queries):

    ring_B_inter = interpolate_ring(ring_B, num_bins=len(ring_A), pad_length=None)
    points = ring_B_inter[:-1]

    center = points.mean(dim=0)

    # Step 2: Compute the angles of each point relative to the center
    # Vector from center to each point
    vectors = points - center

    # Compute the angle of each vector
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])

    # Step 3: Find the point with the angle closest to 0
    # Wrap angles to the range [-pi, pi]
    angles = (angles + 2 * torch.pi) % (2 * torch.pi) - torch.pi

    # Find the index of the minimum angle (closest to 0)
    closest_to_x_axis_idx = torch.argmin(torch.abs(angles))

    aligned_B = torch.roll(ring_B_inter[:-1], shifts=[-closest_to_x_axis_idx], dims=[0])
    aligned_B = torch.cat([aligned_B, aligned_B[0:1]])

    gt_inds = ((aligned_B[:-1].unsqueeze(0) - ring_B[:-1].unsqueeze(1)) ** 2).sum(dim=-1).argmin(dim=1)
    # offsets = ring_B[:-1] - ring_A[gt_inds]
    offsets = ring_B[:-1]
    idx = gt_inds.argmin()
    gt_inds = torch.roll(gt_inds, shifts=[-idx.item()])
    offsets = torch.roll(offsets, shifts=[-idx.item()], dims=[0])

    if len(gt_inds) > num_primitive_queries:
        inds = torch.linspace(0, len(gt_inds) - 1, num_primitive_queries).round().int()
        gt_inds = gt_inds[inds]
        offsets = offsets[inds]
    else:
        gt_inds = F.pad(gt_inds, pad=(0,num_primitive_queries-len(ring_B)+1), value=-1)
        offsets = F.pad(offsets, pad=(0,0,0,num_primitive_queries-len(ring_B)+1), value=-1)

    return aligned_B, gt_inds, offsets

def align_rings_by_roll(ring_A, ring_B, closed=True):
    """
    ring_A: torch (N, 2)
    ring_B: torch (M, 2)
    """
    if closed:
        A = ring_A[:-1]
        B = ring_B[:-1]
    else:
        A = ring_A
        B = ring_B

    if len(B) <= 1:
        return ring_B

    B_rolls = []
    for offset in range(len(B) - 1):
        B_roll = torch.roll(B, shifts=[-offset], dims=[0])
        B_rolls.append(B_roll)

    B_rolls = torch.stack(B_rolls, dim=0)

    dis = ((A.unsqueeze(0) - B_rolls) ** 2).sum(dim=-1) ** 0.5
    roll_idx = dis.sum(dim=1).argmin(dim=0)
    aligned_B = B_rolls[roll_idx.item()]

    if closed:
        aligned_B = torch.cat([aligned_B, aligned_B[0:1]])

    return aligned_B

def sort_rings_by_angle(ring):

    points = ring[:-1]
    center = points.mean(dim=0)

    # Step 2: Compute the angles of each point relative to the center
    # Vector from center to each point
    vectors = points - center

    # Compute the angle of each vector
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])

    # Step 3: Find the point with the angle closest to 0
    # Wrap angles to the range [-pi, pi]
    angles = (angles + 2 * torch.pi) % (2 * torch.pi) - torch.pi

    # Find the index of the minimum angle (closest to 0)
    closest_to_x_axis_idx = torch.argmin(torch.abs(angles))

    sorted_ring = torch.roll(ring[:-1], shifts=[-closest_to_x_axis_idx], dims=[0])
    sorted_ring = torch.cat([sorted_ring, sorted_ring[0:1]])

    return sorted_ring

def cal_angle_for_ring(ring):

    vectors = torch.roll(ring, shifts=[-1], dims=[0]) - ring
    # Compute the angle of each vector
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])

    return angles



def sort_rings_by_y(ring):

    points = ring[:-1]
    pdb.set_trace()
    points[:, 1].min(dim=-1)[1]

    center = points.mean(dim=0)

    # Step 2: Compute the angles of each point relative to the center
    # Vector from center to each point
    vectors = points - center

    # Compute the angle of each vector
    angles = torch.atan2(vectors[:, 1], vectors[:, 0])

    # Step 3: Find the point with the angle closest to 0
    # Wrap angles to the range [-pi, pi]
    angles = (angles + 2 * torch.pi) % (2 * torch.pi) - torch.pi

    # Find the index of the minimum angle (closest to 0)
    closest_to_x_axis_idx = torch.argmin(torch.abs(angles))

    sorted_ring = torch.roll(ring[:-1], shifts=[-closest_to_x_axis_idx], dims=[0])
    sorted_ring = torch.cat([sorted_ring, sorted_ring[0:1]])

    return sorted_ring

    gt_inds = ((aligned_B[:-1].unsqueeze(0) - ring_B[:-1].unsqueeze(1)) ** 2).sum(dim=-1).argmin(dim=1)
    # offsets = ring_B[:-1] - ring_A[gt_inds]
    offsets = ring_B[:-1]
    idx = gt_inds.argmin()
    gt_inds = torch.roll(gt_inds, shifts=[-idx.item()])
    offsets = torch.roll(offsets, shifts=[-idx.item()], dims=[0])

    if len(gt_inds) > num_primitive_queries:
        inds = torch.linspace(0, len(gt_inds) - 1, num_primitive_queries).round().int()
        gt_inds = gt_inds[inds]
        offsets = offsets[inds]
    else:
        gt_inds = F.pad(gt_inds, pad=(0,num_primitive_queries-len(ring_B)+1), value=-1)
        offsets = F.pad(offsets, pad=(0,0,0,num_primitive_queries-len(ring_B)+1), value=-1)

    return aligned_B, gt_inds, offsets


def normalize_rings(rings, eps=1e-8):
    B, Q, N, _ = rings.shape
    boxes = torch.cat([rings.min(dim=2)[0], rings.max(dim=2)[0]], dim=-1)
    widths = (boxes[..., 2] - boxes[..., 0])
    heights = (boxes[..., 3] - boxes[..., 1])
    max_wh = torch.where(widths > heights, widths, heights)

    norm_rings = (rings - boxes.unsqueeze(2)[..., :2]) / (max_wh.unsqueeze(-1).unsqueeze(-1) + eps)
    norm_rings = (norm_rings - 0.5) * 2

    return norm_rings

def get_ring_len(ring):
    if type(ring) == np.ndarray:
        dis = ((ring[:-1] - np.roll(ring[:-1], shift=1, axis=0)) ** 2).sum(axis=-1) ** 0.5
    elif type(ring) == torch.Tensor:
        dis = ((ring[:-1] - torch.roll(ring[:-1], shifts=[1], dims=[0])) ** 2).sum(axis=-1) ** 0.5

    return dis.sum()

def polygonize_cv2(bitmap):

    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [[]]
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]

    return contours

def calculate_polygon_angles(coords, eps=1e-6):
    # Ensure the input is a 2D tensor of shape (N, 2)
    assert coords.dim() == 2 and coords.size(1) == 2

    N = coords.size(0)
    
    # Roll the tensor to get pairs of consecutive points
    coords_next = torch.roll(coords, shifts=-1, dims=0)
    coords_prev = torch.roll(coords, shifts=1, dims=0)

    # Calculate vectors from each point to the next and previous points
    vectors_next = coords_next - coords
    vectors_prev = coords_prev - coords

    valid_mask = (vectors_next.abs() > eps).any(dim=-1)
    valid_mask = valid_mask & (vectors_prev.abs() > eps).any(dim=-1)

    """
    if (torch.norm(vectors_next, dim=1, keepdim=True) < eps).any():
        pdb.set_trace()
    if (torch.norm(vectors_prev, dim=1, keepdim=True) < eps).any():
        pdb.set_trace()
    """

    # Normalize the vectors
    norm_vectors_next = vectors_next / (torch.norm(vectors_next, dim=1, keepdim=True))
    norm_vectors_prev = vectors_prev / (torch.norm(vectors_prev, dim=1, keepdim=True))

    # Calculate dot products between consecutive vectors
    dot_products = torch.sum(norm_vectors_next * norm_vectors_prev, dim=1)

    # Ensure the values are within the valid range for arccos [-1, 1] to prevent NaNs
    dot_products = torch.clamp(dot_products, -1.0 + eps, 1.0 - eps)

    # Calculate the angles using arccos (in radians)
    angles = torch.acos(dot_products)

    # Convert angles to degrees if needed
    # angles_degrees = torch.rad2deg(angles)

    return angles, valid_mask

def scores_to_permutations(scores, ignore_thre=0):
    """
    Input a batched array of scores and returns the hungarian optimized 
    permutation matrices.
    """
    B, N, N = scores.shape

    scores = scores.detach().cpu().numpy()
    perm = np.zeros_like(scores)
    for b in range(B):
        if ignore_thre is not None:
            valid_rows = (scores[b] > ignore_thre).any(axis=1)
            # valid_cols = (scores[b] > 0).any(axis=0)
            valid_scores = scores[b][valid_rows]
            # assert (valid_rows == valid_cols).all()
            r, c = linear_sum_assignment(-scores[b, valid_rows][:, valid_rows])
            r = valid_rows.nonzero()[0][r]
            c = valid_rows.nonzero()[0][c]

        else:
            r, c = linear_sum_assignment(-scores[b])

        perm[b,r,c] = 1
    return torch.tensor(perm)

def permutations_to_polygons(perm, graph, out='torch', ignore_thre=0, min_poly_size=4,
                             return_first_contour=True):
    B, N, N = perm.shape

    def bubble_merge(poly):
        s = 0
        P = len(poly)
        while s < P:
            head = poly[s][-1]

            t = s+1
            while t < P:
                tail = poly[t][0]
                if head == tail:
                    poly[s] = poly[s] + poly[t][1:]
                    del poly[t]
                    poly = bubble_merge(poly)
                    P = len(poly)
                t += 1
            s += 1
        return poly

    diag = torch.logical_not(perm[:,range(N),range(N)])
    batch = []
    for b in range(B):
        b_perm = perm[b]
        b_graph = graph[b]
        b_diag = diag[b]

        idx = torch.arange(N)[b_diag]

        if idx.shape[0] > 0:
            # If there are vertices in the batch

            b_perm = b_perm[idx,:]
            b_graph = b_graph[idx,:]
            b_perm = b_perm[:,idx]

            first = torch.arange(idx.shape[0]).unsqueeze(1)
            second = torch.argmax(b_perm, dim=1).unsqueeze(1).cpu()
            if ignore_thre is not None:
                valid_rows = (b_perm > ignore_thre).any(dim=1)

                first = first[valid_rows]
                second = second[valid_rows]

            polygons_idx = torch.cat((first, second), dim=1).tolist()
            polygons_idx = bubble_merge(polygons_idx)

            batch_poly = []
            for p_idx in polygons_idx:
                if len(p_idx) < min_poly_size + 1:
                    continue

                if out == 'torch':
                    batch_poly.append(b_graph[p_idx,:])
                elif out == 'numpy':
                    batch_poly.append(b_graph[p_idx,:].cpu().numpy())
                elif out == 'list':
                    g = b_graph[p_idx,:] * 300 / 320
                    g[:,0] = -g[:,0]
                    g = torch.fliplr(g)
                    batch_poly.append(g.tolist())
                elif out == 'coco':
                    g = b_graph[p_idx,:] * 300 / 320
                    g = torch.fliplr(g)
                    batch_poly.append(g.view(-1).tolist())
                else:
                    print("Indicate a valid output polygon format")
                    exit()
                if return_first_contour and len(batch) > 0:
                    break
            batch.append(batch_poly)

        else:
            # If the batch has no vertices
            batch.append([])

    return batch


def decode_ring_next(next_idxes, valid_mask=None, min_dis=2, points=None):
    x = 0
    pred_idxes = []
    while(next_idxes[x] > x and (valid_mask is None or valid_mask[x])):
        pred_idxes.append(x)
        x = next_idxes[x]

    if (valid_mask is None or valid_mask[x]) and (points is None or (len(points) - x) >= min_dis):
        pred_idxes.append(x)

    pred_idxes = torch.tensor(pred_idxes).long()

    return pred_idxes

def binarize(values, num_bins, min_v, max_v):
    bins = torch.linspace(min_v, max_v, num_bins)
    discretized_values = torch.bucketize(values, bins) - 1
    discretized_values = torch.clamp(discretized_values, 0, num_bins-1)
    return discretized_values

def post_process_without_format(pred_rings, ring_pred_next, all_idxes, all_ring_sizes, length,
                                stride, decode_type='next', device='cpu', num_max_points=512):

    def post_process_fun(pred_rings, ring_pred_next, all_idxes, all_ring_sizes):

        pred_polygons = []
        for i in range(len(all_ring_sizes)):
            cur_pred_polygon = []
            for j in range(len(all_ring_sizes[i])):
                cur_mask = (all_idxes[:,0] == i) & (all_idxes[:,1] == j)
                cur_rings = pred_rings[cur_mask]
                cur_ring_len = all_ring_sizes[i][j]

                cur_pred_ring = torch.zeros(cur_ring_len, 2)
                cur_count = torch.zeros(cur_ring_len)

                if ring_pred_next is not None:
                    cur_ring_next = ring_pred_next[cur_mask]
                    ring_next = torch.zeros(cur_ring_len, cur_ring_len)

                # if cur_rings.shape[0] > 1:
                #     pdb.set_trace()
                for k in range(cur_rings.shape[0]):
                    cur_valid_mask = (cur_rings[k] >= 0).all(dim=1)
                    temp = (torch.arange(length) + stride * k) % cur_ring_len
                    cur_pred_ring[temp[cur_valid_mask]] += cur_rings[k][cur_valid_mask]
                    cur_count[temp] += 1
                    if ring_pred_next is not None:
                        ring_next[temp[cur_valid_mask,None], temp[cur_valid_mask]] += cur_ring_next[k, cur_valid_mask][:, cur_valid_mask]

                temp = cur_pred_ring[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)
                if len(temp) > num_max_points:
                    sample_idxes = torch.linspace(0, len(temp)-1, num_max_points).round().long()
                    temp = temp[sample_idxes]

                if decode_type == 'next':
                    pred_idxes = decode_ring_next(ring_next.max(dim=1)[1], points=None)
                elif decode_type == 'dp':
                    pred_idxes = decode_ring_dp(
                        points=temp,
                        scores=ring_next if ring_pred_next is not None else None,
                        device=device
                    )
                    if pred_idxes is not None:
                        pred_idxes = pred_idxes.cpu()
                elif decode_type == 'greedy':
                    pred_idxes = decode_ring_greedy(ring_next, points=temp)
                elif decode_type == 'none':
                    pred_idxes = torch.arange(len(temp))

                cur_pred_polygon.append(temp[pred_idxes])

            pred_polygons.append(cur_pred_polygon)

        # format_results(pred_polygons, img_meta=img_meta)
        return pred_polygons

    pred_polygons = post_process_fun(pred_rings, ring_pred_next, all_idxes, all_ring_sizes)
    return pred_polygons

def collect_rings(poly_preds):
    B, N, _ = poly_preds.shape

def cal_pairwise_areas(points, device='cpu', eps=1e-8):

    def reflect_vector(a, b):
        a_norm = a / torch.norm(a, dim=1, keepdim=True)
        proj_b_on_a = (torch.sum(b * a_norm, dim=1, keepdim=True)) * a_norm
        c = 2 * proj_b_on_a - b

        return c

    def get_trapezoid_area(points):
        # points: (B,N,2)

        ring = torch.cat([points, points[:, :1]], dim=1)
        x = ring[:, :, 0]
        y = ring[:, :, 1]
        # area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
        A = y[:, :-1] + y[:, 1:]
        B = x[:, 1:] - x[:, :-1]

        areas = 0.5 * A * B # area of each small trapezoid

        return areas

    points = points.to(device)
    N = points.shape[0]
    # areas = areas.unsqueeze(0).unsqueeze(0).repeat(N,N,1)
    triu = torch.triu(torch.ones(N,N, dtype=torch.bool, device=device), diagonal=1).T
    V3_mask = torch.bitwise_xor(triu.unsqueeze(1), triu.unsqueeze(0))
    V3_mask[triu] = V3_mask[triu].bitwise_xor(True)
    # temp = (areas * mask).sum(dim=-1)

    """
    areas = 0.5 * A * B # area of each small trapezoid
    accu_areas = torch.cumsum(areas, dim=0)
    accu_areas = torch.cat([torch.zeros(1), accu_areas])

    rows, cols = torch.meshgrid(torch.arange(N), torch.arange(N))
    inds = torch.stack([rows, cols], dim=-1)

    int_area = accu_areas[inds[:,:,1]] - accu_areas[inds[:,:,0]]
    int_area[inds[:,:,1] < inds[:,:,0]] += accu_areas[-1]
    """

    V2 = torch.cat([
        points.unsqueeze(1).repeat(1,N,1),
        points.unsqueeze(0).repeat(N,1,1)
    ], dim=-1)
    V3 = torch.cat([
        points.view(N,1,1,2).repeat(1,N,N,1),
        points.view(1,N,1,2).repeat(N,1,N,1),
        points.view(1,1,N,2).repeat(N,N,1,1),
    ], dim=-1)

    x1 = V3[..., 2:4] - V3[..., :2]
    x2 = V3[..., 4:] - V3[..., :2]
    # V3_dis = (x1[..., 0] * x2[..., 1] - x1[..., 1] * x2[..., 0]).abs() # cross-product, need to add minus since the direction of y-aixs is different in an image
    dot_x12 = (x1 * x2).sum(dim=-1)
    V3_dis = torch.norm((x2 - dot_x12.unsqueeze(-1) / ((torch.norm(x1, dim=-1).unsqueeze(-1) ** 2) + eps) * x1), dim=-1)
    cost = (V3_dis * V3_mask).sum(dim=-1)

    # V3_refl = reflect_vector(x1.view(-1,2), x2.view(-1,2)).view(N,N,N,2) + V3[..., :2]
    # V3_dir = - (x1[..., 0] * x2[..., 1] - x1[..., 1] * x2[..., 0]) # cross-product, need to add minus since the direction of y-aixs is different in an image

    """
    V3_area = get_trapezoid_area(V3[..., 4:].view(N*N, N, 2)).view(N,N,N)
    V3_refl_area = get_trapezoid_area(V3_refl.view(N*N, N, 2)).view(N,N,N)

    V2_area = 0.5 * (V2[...,1] + V2[...,3]) * (V2[...,2] - V2[...,0])
    V3_abs_area = (torch.where(V3_dir >= 0, V3_area, V3_refl_area) * V3_mask).sum(dim=-1)
    V3_abs_area2 = (V3_area * V3_mask).sum(dim=-1)

    tot_area = (V3_abs_area - V2_area).abs()
    tot_area2 = (V3_abs_area2 - V2_area).abs()
    pdb.set_trace()
    cost = tot_area
    """

    arange_N = torch.arange(N, device=device)
    shift_idxes = (arange_N.unsqueeze(0) + arange_N.unsqueeze(1)) % N
    rows, cols = torch.meshgrid(arange_N, arange_N)
    cost = cost[rows, shift_idxes]

    return cost

def cal_pairwise_areas_old(points, device='cpu'):

    N = points.shape[0]
    ring = torch.cat([points, points[:1]])
    x = ring[:, 0]
    y = ring[:, 1]
    # area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    A = y[:-1] + y[1:]
    B = x[1:] - x[:-1]
    arange_N = torch.arange(N, device=device)

    areas = 0.5 * A * B # area of each small trapezoid

    accu_areas = torch.cumsum(areas, dim=0)
    accu_areas = torch.cat([torch.zeros(1, device=device), accu_areas])

    rows, cols = torch.meshgrid(arange_N, arange_N)
    inds = torch.stack([rows, cols], dim=-1)

    int_area = accu_areas[inds[:,:,1]] - accu_areas[inds[:,:,0]]
    int_area[inds[:,:,1] < inds[:,:,0]] += accu_areas[-1]

    V = torch.cat([points.unsqueeze(1).repeat(1,N,1), points.unsqueeze(0).repeat(N,1,1)], dim=-1)
    pair_area = 0.5 * (V[...,1] + V[...,3]) * (V[...,2] - V[...,0])

    tot_area = (int_area - pair_area).abs()

    shift_idxes = (arange_N.unsqueeze(0) + arange_N.unsqueeze(1)) % N
    cost = tot_area[rows, shift_idxes]

    return cost

def decode_ring_dp(points, scores=None, step_size=1, lam=5, max_step_size=10, device='cpu'):
    points = points.to(device)

    import time
    t0 = time.time()
    N = points.shape[0]
    inf = 1e8
    dp = torch.zeros(N, N, device=device) + inf
    fa = torch.zeros(N, N, dtype=torch.long, device=device)

    cost_areas = cal_pairwise_areas(points, device=device)
    # cost_areas = cal_pairwise_areas_old(points)
    norm_cost = torch.zeros(N,N, device=device) + lam
    cost = cost_areas + norm_cost

    if scores is not None:
        scores = scores.to(device)
        norm_cost = (1 - scores.sigmoid()) * lam
        cost += norm_cost

    t1 = time.time()

    dp[:,0] = 0
    dp[:,1] = cost[:,1]
    fa[:,1] = 1

    N_ks = torch.arange(1, min(N, max_step_size) + 1, step_size, device=device)
    arange_N = torch.arange(N, device=device).unsqueeze(0)
    for l in range(2, N, step_size):
        # ks = torch.arange(1, min(max_step_size, l) + 1, step_size, device=device)
        ks = N_ks[:l]
        # rows = (torch.arange(N, device=device).unsqueeze(0) + ks.unsqueeze(1)) % N
        rows = (arange_N + ks.unsqueeze(1)) % N
        cols = l - ks.view(-1,1).repeat(1,N)

        new_cost = cost[:, ks] + dp[rows.view(-1), cols.view(-1)].view(len(ks), N).permute(1,0)
        min_new_cost, min_new_cost_idxes = new_cost.min(dim=1)

        dp[:,l] = min_new_cost
        fa[:,l] = ks[min_new_cost_idxes]

    t2 = time.time()

    cur_idx = dp[:,N-1].argmin().item()
    k = N-1
    idxes = [cur_idx]
    fa = fa.cpu()
    while k > 1:
        next_idx = (cur_idx + fa[cur_idx, k]) % N
        k = k - fa[cur_idx, k].item()
        cur_idx = next_idx

        idxes.append(cur_idx.item())
    t3 = time.time()

    if len(idxes) < 3:
        return None

    if N > 100:
        print(f'dp time: {N} {t1-t0} {t2-t1} {t3-t2}')

    return torch.tensor(idxes, device=device)

def decode_ring_greedy(scores, points, step_size=2):
    N = scores.shape[0]
    sorted_idxes = scores.view(-1).argsort(descending=True)
    rows, cols = sorted_idxes // N, sorted_idxes % N

def decode_poly_jsons(poly_json, scale=1, step_size=8, device='cpu', results_format='coco'):

    simp_coords = []
    coords_list = poly_json['coordinates']
    for coords in coords_list:
        scaled_coords = torch.tensor(coords, device=device) * scale
        scaled_coords = interpolate_ring(
            scaled_coords,
            interval=step_size,
            device=device
        )
        pred_idxes = decode_ring_dp(scaled_coords)
        if pred_idxes is not None:
            if results_format == 'coco':
                simp_coord = scaled_coords[pred_idxes].view(-1).tolist()
            elif results_format == 'json':
                simp_coord = scaled_coords[pred_idxes].tolist()
            else:
                raise ValueError
            simp_coords.append(simp_coord)

    if len(simp_coords) == 0:
        simp_coords = [[]]

    if results_format == 'coco':
        return simp_coords
    elif results_format == 'json':
        return dict(
            type='Polygon',
            coordinates=simp_coords
        )
    else:
        raise ValueError

def poly_json2coco(poly_json, scale=1.):
    coords_list = poly_json['coordinates']
    new_coords_list = []
    for coords in coords_list:
        new_coords = (torch.tensor(coords) * scale).view(-1).tolist()
        new_coords_list.append(new_coords)

    return new_coords_list


def worker(args):
    imgs, offset, clockwise, scale = args
    cur_shapes = shapes(imgs, mask=imgs > 0)
    all_coords = []
    for shape, value in cur_shapes:
        coords = shape['coordinates']
        scaled_coords = []
        for x in coords:
            x = (np.array(x) + offset) * scale
            if clockwise:
                x = x[::-1]
                # x = x.flip(dims=[0])
            scaled_coords.append(x.tolist())

        all_coords.extend(scaled_coords)

    polygon = dict(
        type='Polygon',
        coordinates=all_coords
    )
    return polygon

def collect_result(result):
    global results
    results.extend(result)

def separate_ring(ring, crop_len, stride, array_type='torch', pad_value=-1):

    N, _ = ring.shape
    if N < crop_len:
        if array_type == 'numpy':
            ring = np.concatenate([ring, np.zeros((crop_len-N, 2)) + pad_value])
        else:
            ring = torch.cat([ring, torch.zeros(crop_len-N, 2) + pad_value])

        return [ring]

    if array_type == 'numpy':
        # repeated_ring = np.concatenate([ring[:-1], ring], axis=0)
        repeated_ring = np.concatenate([ring, ring], axis=0)
    else:
        # repeated_ring = torch.cat([ring[:-1], ring], dim=0)
        repeated_ring = torch.cat([ring, ring], dim=0)

    num_parts = math.ceil((N - crop_len) / stride) \
            if math.ceil((N - crop_len) / stride) * stride + crop_len >= N \
            else math.ceil((N - crop_len) / stride) + 1

    if array_type == 'numpy':
        idxes = np.arange(num_parts + 1)  * stride
    else:
        idxes = torch.arange(num_parts + 1)  * stride

    rings = [repeated_ring[x:x + crop_len] for x in idxes]
    return rings

def sample_segments_from_json(polygons, seg_len=50, stride=25, array_type='torch', **kwargs):
    """
    polygons: List of json dicts of polygons
    interval: sampling distance in each ring of the polygons
    seg_len: maximum length of each separated segment sequences
    stride: stride to sample segment sequences in each ring of the polygons
    """

    sampled_segs = []
    sampled_seg_sizes = []
    poly2segs_idxes = []
    segs2poly_idxes = []
    array_fun = torch.tensor if array_type == 'torch' else np.array
    seg_cnt = 0

    # for i, polygon in tqdm(enumerate(polygons), desc='sampling rings...'):
    for i, polygon in enumerate(polygons):
        rings = polygon['coordinates']
        sizes = []
        cur_poly2segs_idxes = []

        assert len(rings) > 0, 'Empty polygons are not allowed when sampling segment sequences!'
        for j, ring in enumerate(rings):
            sampled_ring = interpolate_ring(array_fun(ring), type=array_type, **kwargs)[:-1]
            ring_parts = separate_ring(sampled_ring, crop_len=seg_len, stride=stride)

            idx1 = np.array([i] * len(ring_parts))
            idx2 = np.array([j] * len(ring_parts))
            idx = np.stack([idx1, idx2], axis=1)
            segs2poly_idxes.append(idx)

            sampled_segs.append(ring_parts)
            cur_poly2segs_idxes.append((torch.arange(len(ring_parts)) + seg_cnt).tolist())
            sizes.append(len(sampled_ring))
            seg_cnt += len(ring_parts)

        sampled_seg_sizes.append(sizes)
        poly2segs_idxes.append(cur_poly2segs_idxes)

    sampled_segs = np.concatenate(sampled_segs, axis=0)
    segs2poly_idxes = np.concatenate(segs2poly_idxes, axis=0)

    return sampled_segs, sampled_seg_sizes, poly2segs_idxes, segs2poly_idxes


def assemble_segments(segments, seg_idxes, seg_sizes, length=50, stride=30, max_len=512, **kwargs):
    rings = []
    poly2ring_idxes = []
    ring_cnt = 0
    others = {key: [] for key in kwargs.keys()}

    for i in range(len(seg_idxes)):
        cur_pred_polygon = []
        cur_poly2ring_idxes = []
        for j in range(len(seg_idxes[i])):
            cur_segs_len = seg_sizes[i][j]
            cur_ring = torch.zeros(cur_segs_len, 2)
            cur_ring_cnts = torch.zeros(cur_segs_len)
            cur_others = dict()
            for key in kwargs.keys():
                cur_others[key] = torch.zeros(cur_segs_len, *kwargs[key].shape[2:])

            for k, idx in enumerate(seg_idxes[i][j]):
                cur_segments = segments[idx]
                cur_seg_mask = (cur_segments >= 0).all(dim=1)
                # if filter_minus else torch.ones(len(cur_segments), dtype=torch.bool)

                cur_segs_loc = (torch.arange(length) + stride * k) % cur_segs_len
                cur_ring[cur_segs_loc[cur_seg_mask]] += cur_segments[cur_seg_mask]
                cur_ring_cnts[cur_segs_loc[cur_seg_mask]] += 1

                for key in kwargs.keys():
                    cur_seg_others = kwargs[key][idx]
                    cur_others[key][cur_segs_loc[cur_seg_mask]] += cur_seg_others[cur_seg_mask]

            cur_ring = cur_ring[cur_ring_cnts > 0] / cur_ring_cnts[cur_ring_cnts > 0].unsqueeze(1)
            for key in kwargs.keys():
                cur_others[key] = cur_others[key][cur_ring_cnts > 0] / \
                        cur_ring_cnts[cur_ring_cnts > 0].view(-1, *([1] * (len(cur_others[key].shape)-1)))

            if len(cur_ring) > max_len:
                sample_idxes = torch.linspace(0, len(cur_ring)-1, max_len).round().long()
                cur_ring = cur_ring[sample_idxes]
                for key in kwargs.keys():
                    cur_others[key] = cur_others[key][sample_idxes]

            rings.append(cur_ring)
            cur_poly2ring_idxes.append(ring_cnt)
            for key in kwargs.keys():
                others[key].append(cur_others[key])
            ring_cnt += 1

        poly2ring_idxes.append(cur_poly2ring_idxes)

    return rings, poly2ring_idxes, others

def assemble_rings(rings, ring_idxes, format='coco'):
    polygons = []
    for i in range(len(ring_idxes)):
        cur_idxes = ring_idxes[i]
        cur_polygon = []
        for j in range(len(cur_idxes)):
            cur_ring = rings[ring_idxes[i][j]]
            if format == 'coco':
                cur_ring = cur_ring.view(-1).tolist()
            elif format == 'json':
                cur_ring = cur_ring.view(-1,2).tolist()

            cur_polygon.append(cur_ring)

        if format == 'json':
            cur_polygon = dict(
                type='Polygon',
                coordinates=cur_polygon
            )
        polygons.append(cur_polygon)

    return polygons


def cal_pairwise_dis(points, sizes, device='cpu', eps=1e-8, max_step_size=20, ref_points=None):

    points = points.to(device)
    B, N = points.shape[:2]
    K = min(sizes.max(), max_step_size)
    if ref_points is not None:
        ref_points = ref_points.to(device)

    triu = torch.triu(torch.ones(K,K, dtype=torch.bool, device=device), diagonal=1).T
    V3_mask = triu.unsqueeze(0)

    arange_N = torch.arange(N, device=device)
    arange_K = torch.arange(K, device=device)
    I3 = torch.cat([
        arange_N.view(N,1,1,1).repeat(1,K,K,1),
        arange_K.view(1,K,1,1).repeat(N,1,K,1),
        arange_K.view(1,1,K,1).repeat(N,K,1,1),
    ], dim=-1)
    I3[..., 1:] += arange_N.view(N,1,1,1)

    I3 = I3.unsqueeze(0).repeat(B,1,1,1,1)
    I3[..., 1:] %= sizes.view(B,1,1,1,1)

    V3 = torch.gather(points,1,I3.view(B,-1,1).repeat(1,1,2)).view(B,N,K,K,6)
    if ref_points is not None:
        V3_ref = torch.gather(ref_points,1,I3.view(B,-1,1).repeat(1,1,2)).view(B,N,K,K,6)
        V3[..., :4] = V3_ref[..., :4]

    # I3[..., 1:] %= size
    # V3 = points[I3].view(N,K,K,-1)

    x1 = V3[..., 2:4] - V3[..., :2]
    x2 = V3[..., 4:] - V3[..., :2]
    dot_x12 = (x1 * x2).sum(dim=-1)
    V3_dis = torch.norm((x2 - dot_x12.unsqueeze(-1) / ((torch.norm(x1, dim=-1).unsqueeze(-1) ** 2) + eps) * x1), dim=-1)
    cost = (V3_dis * V3_mask).sum(dim=-1)

    new_cost = torch.ones(B, N, N, device=device) * 1e8
    inf_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    inf_mask[arange_N.unsqueeze(0) < sizes.unsqueeze(1)] = True
    A = inf_mask.unsqueeze(2).repeat(1,1,N)
    B = A.transpose(1,2)
    inf_mask = (A & B)[:,:,:K]
    cost[~inf_mask] = 1e8

    new_cost[:,:,:K] = cost

    return new_cost

def batchify(sizes):

    sorted_idxes = np.argsort(np.array(sizes))
    base_size = 512 * 512 * 4

    batch_idx_list = []
    batch_size_list = []

    cur_bin = []
    cur_sizes = []
    cur_size = 0
    cur_cnt = 0
    for i, idx in enumerate(sorted_idxes):
        if sizes[idx] < 4:
            break
        if (cur_cnt + 1) * sizes[idx] ** 2 <= base_size:
            cur_bin.append(idx)
            cur_sizes.append(sizes[idx])
            cur_cnt += 1
        else:
            batch_idx_list.append(cur_bin)
            batch_size_list.append(cur_sizes)
            cur_bin = [idx]
            cur_cnt = 1
            cur_sizes = [sizes[idx]]

    if len(cur_bin) > 0:
        batch_idx_list.append(cur_bin)
        batch_size_list.append(cur_sizes)

    return batch_idx_list, batch_size_list


def save_polygons(polygons, transform, crs, out_path, upscale=1):

    if len(polygons) == 0:
        return None
    """Place holder to format result to dataset specific output."""

    offset = np.array([0,0]).reshape(1,2)
    global_polygons = []

    for polygon in polygons:
        new_rings = []
        for ring in polygon['coordinates']:
            ring = np.array(ring)
            new_ring = np.stack((transform * (ring * upscale + offset).transpose(1,0)), axis=1)
            if len(new_ring) >= 4:
                new_rings.append(new_ring)

        if len(new_rings) > 0:
            new_polygon = shapely.geometry.Polygon(new_rings[0], new_rings[1:] if len(new_rings) > 1 else None)
            global_polygons.append(new_polygon)

    gdf = gpd.GeoDataFrame(geometry=global_polygons)
    gdf.crs = crs
    gdf.to_file(out_path)

def pad_longest_cat(seq_list):
    N = max([len(seq) for seq in seq_list])
    new_seq_list = []
    for seq in seq_list:
        if len(seq) < N:
            new_seq = F.pad(seq, pad=[0,0,0,N-len(seq)], value=-1)
        else:
            new_seq = seq

        new_seq_list.append(new_seq)

    new_seq = torch.stack(new_seq_list, dim=0)
    return new_seq


def batch_decode_ring_dp(batch_points, sizes, step_size=1, lam=5, max_step_size=20, device='cpu',
                         ref_rings=None, only_return_dp=False, result_device='cpu', return_both=False):

    K = min(sizes.max().item(), max_step_size)

    B, N, _ = batch_points.shape
    inf = 1e8
    dp = torch.zeros(B,N,N+1, device=device) + inf
    fa = torch.zeros(B,N,N+1, dtype=torch.long, device=device)
    cost = torch.zeros(B,N,N+1, device=device) + inf

    cost_areas = cal_pairwise_dis(batch_points, sizes, device=device, max_step_size=max_step_size,
                                  ref_points=ref_rings)

    norm_cost = torch.zeros(B,N,N, device=device) + lam
    cost[:,:,:N] = cost_areas + norm_cost

    dp[:,:,0] = 0
    dp[:,:,1] = cost[:,:,1]
    fa[:,:,1] = 1

    N_ks = torch.arange(1, min(K, max_step_size) + 1, step_size, device=device)
    arange_N = torch.arange(N, device=device).unsqueeze(0)
    for l in range(2, N+1, step_size):
        ks = N_ks[:l]
        rows = (arange_N + ks.unsqueeze(1)) % N
        cols = l - ks.view(-1,1).repeat(1,N)

        new_cost = cost[:,:,ks] + dp[:, rows.view(-1), cols.view(-1)].view(B,len(ks), N).permute(0,2,1)
        min_new_cost, min_new_cost_idxes = new_cost.min(dim=-1)

        dp[:,:,l] = min_new_cost
        fa[:,:,l] = ks[min_new_cost_idxes]

    if only_return_dp:
        return dp


    batch_dp_points = []
    batch_idxes = []
    # cur_idxes = dp[torch.arange(B), :, sizes-1].argmin(dim=1).to(result_device)
    cur_idxes = dp[torch.arange(B), :, sizes].argmin(dim=1).to(result_device)
    batch_idxes.append(cur_idxes)

    sizes = sizes.to(result_device)
    fa = fa.to(result_device)
    arange_B = torch.arange(B)
    # ks = (sizes - 1).to(result_device)
    ks = sizes.to(result_device)
    length = torch.ones(B, dtype=torch.long, device=result_device)
    batch_points = batch_points.to(result_device)

    while((ks > 1).any()):
        temp = fa[arange_B, cur_idxes, ks]
        next_idxes = (cur_idxes + temp) % sizes
        ks = ks - temp
        length[cur_idxes != next_idxes] += 1
        cur_idxes = next_idxes
        batch_idxes.append(cur_idxes)

    batch_idxes = torch.stack(batch_idxes, dim=1)
    dp_points = torch.gather(batch_points, 1, batch_idxes.unsqueeze(-1).repeat(1,1,2))
    dp_points = [x[:y] for x, y in zip(dp_points, length)]

    if return_both:
        return dp, dp_points

    return dp_points


def simplify_rings_dp(rings, max_step_size=50, lam=5, device=None, ref_rings=None,
                      only_return_dp=False, drop_last=True):

    if device is None:
        device = rings[0].device

    len_rings = [len(x) for x in rings]
    batch_idx_list, batch_size_list = batchify(len_rings)

    new_all_rings = [None] * len(rings)
    # for i in tqdm(range(len(batch_idx_list))):
    dps = []
    for i in range(len(batch_idx_list)):
        idxes = batch_idx_list[i]

        if drop_last:
            sizes = torch.tensor(batch_size_list[i], device=device) - 1
            cur_rings = [rings[x][:-1] for x in idxes]
        else:
            sizes = torch.tensor(batch_size_list[i], device=device)
            cur_rings = [rings[x] for x in idxes]

        cur_rings = pad_longest_cat(cur_rings)

        if ref_rings is not None:
            cur_ref_rings = [ref_rings[x][:-1] for x in idxes]
            cur_ref_rings = pad_longest_cat(cur_ref_rings)
        else:
            cur_ref_rings = None

        results = batch_decode_ring_dp(
            cur_rings, sizes, max_step_size=max_step_size, lam=lam,
            device=device, ref_rings=cur_ref_rings,
            only_return_dp=only_return_dp
        )
        if only_return_dp:
            dp = results
            dps.append(dp)
        else:
            decoded_rings = results
            for i, idx in enumerate(idxes):
                new_all_rings[idx] = decoded_rings[i] if len(decoded_rings[i]) >= 3 else rings[idx][:-1].cpu()

    if only_return_dp:
        return dps
    else:
        return new_all_rings

# def sample_rings_from_json(polygons, interval=2, num_max_lens=512, num_min_lens=8, array_type='torch', only_exterior=False):
def sample_rings_from_json(polygons, array_type='torch', only_exterior=False, **kwargs):
    """
    polygons: List of json dicts of polygons
    interval: sampling distance in each ring of the polygons
    seg_len: maximum length of each separated segment sequences
    stride: stride to sample segment sequences in each ring of the polygons
    """

    sampled_rings = []
    sampled_ring_sizes = []
    poly2ring_idxes = []
    array_fun = torch.tensor if array_type == 'torch' else np.array
    ring_cnt = 0

    # for i, polygon in tqdm(enumerate(polygons), desc='sampling rings...'):
    for i, polygon in enumerate(polygons):
        rings = polygon['coordinates']
        sizes = []
        cur_poly2ring_idxes = []

        assert len(rings) > 0, 'Empty polygons are not allowed when sampling segment sequences!'
        if only_exterior:
            rings = [rings[0]]

        for j, ring in enumerate(rings):
            ring = array_fun(ring)
            if (ring > 0).sum() > 0:
                sampled_ring = interpolate_ring(
                    ring, type=array_type, drop_last=False, **kwargs
                )
            else:
                sampled_ring = ring

            sampled_rings.append(sampled_ring)
            cur_poly2ring_idxes.append(ring_cnt)
            ring_cnt += 1

        sampled_ring_sizes.append(sizes)
        poly2ring_idxes.append(cur_poly2ring_idxes)

    # sampled_segs = np.concatenate(sampled_rings, axis=0)

    return sampled_rings, sampled_ring_sizes, poly2ring_idxes



def simplify_poly_jsons(poly_jsons, lam=4, max_step_size=80, device='cpu', scale=1., format='coco',
                        verbose=False, **kwargs):

    all_rings, ring_sizes, poly2ring_idxes = sample_rings_from_json(poly_jsons, **kwargs)

    all_rings = [x.to(device) * scale for x in all_rings]

    len_rings = [len(x) for x in all_rings]
    batch_idx_list, batch_size_list = batchify(len_rings)
    t2 = time.time()

    new_all_rings = [None] * len(all_rings)
    iterator = range(len(batch_idx_list)) if not verbose else tqdm(range(len(batch_idx_list)))
    for i in iterator:
        idxes = batch_idx_list[i]
        sizes = batch_size_list[i]

        cur_rings = [all_rings[x][:-1] for x in idxes]
        cur_rings = pad_longest_cat(cur_rings)
        cur_rings = cur_rings.cuda()
        sizes = torch.tensor(sizes).cuda() - 1
        # pdb.set_trace()
        decoded_rings = batch_decode_ring_dp(
            cur_rings, sizes, max_step_size=max_step_size, lam=lam, device=cur_rings.device
        )
        cur_rings = cur_rings.cpu()

        for j, idx in enumerate(idxes):
            new_all_rings[idx] = decoded_rings[j] if len(decoded_rings[j]) >= 4 else all_rings[idx][:-1]

    simp_polygons = assemble_rings(new_all_rings, poly2ring_idxes, format=format)

    return simp_polygons

def sample_segments_from_rings(rings, max_len):
    new_rings = []
    is_complete = []
    for ring in rings:
        K = len(ring[:-1])
        shift = np.random.randint(0, K - 1)
        shifted_ring = torch.roll(ring[:-1], shifts=[shift], dims=[0])
        new_ring = torch.zeros(max_len, 2, device=ring.device) - 1
        new_ring[:K] = shifted_ring[:max_len]
        new_rings.append(new_ring)
        is_complete.append(K <= max_len)

    new_rings = torch.stack(new_rings, dim=0)
    is_complete = torch.tensor(is_complete, device=rings[0].device)

    return new_rings, is_complete

def cal_winding_number(rings, m, c=1, eps=1e-6):
    """
    Paper: MonteFloor: Extending MCTS for Reconstructing Accurate Large-Scale Floor Plans
    Eq. (10)
    """
    rings.flip(dims=[0])

    u = rings
    v = torch.roll(rings, dims=[0], shifts=[-1])

    um = m.unsqueeze(0) - u.unsqueeze(1)
    vm = m.unsqueeze(0) - v.unsqueeze(1)

    det = um[..., 0] * vm[..., 1] - um[..., 1] * vm[..., 0]
    w = c * det / (1 + torch.abs(c * det))

    dot = (um * vm).sum(dim=-1) / (torch.norm(um, dim=-1) * torch.norm(vm, dim=-1) + 1e-8)
    angle_uvm = torch.arccos(dot.clamp(-1+eps, 1-eps))

    wn = (angle_uvm * w).sum(dim=0)
    wn = wn / (torch.pi * 2)

    return wn

