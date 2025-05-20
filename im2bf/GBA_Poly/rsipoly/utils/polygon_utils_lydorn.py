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
    projected_pred_contours = project_onto_geometry(sampled_pred_contours, gt_contours)
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
        return np.array([]), [], []
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
            if min_precision < pred_polygon.intersection(gt_polygon).area / pred_polygon.area:
                filtered_polygons.append([pred_polygon, gt_polygon])
                break


    # gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons.geoms for contour in [polygon.exterior, *polygon.interiors]])
    gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons for contour in [polygon.exterior, *polygon.interiors]])
    half_tangent_max_angles = [compute_contour_measure(pred_polygon, gt_polygon, sampling_spacing=sampling_spacing, max_stretch=max_stretch, metric_name=metric_name)
                               for pred_polygon, gt_polygon in tqdm(filtered_polygons, desc='Computing MTAs')]
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

        for i, polygon in tqdm(enumerate(polygons), desc='sampling rings...'):
            # rings = np.array(polygon['coordinates'])
            rings = polygon['coordinates']
            sizes = []
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

def compute_IoU_cIoU(coco, coco_gti):

    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        N = 0
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            # m = np.mod(m.sum(axis=-1), 2)
            m = m.sum(axis=-1)

            if _idx == 0:
                mask = m.reshape((img['height'], img['width']))
                N = len(annotation['segmentation'][0]) // 2
            else:
                mask = mask + m.reshape((img['height'], img['width']))
                N = N + len(annotation['segmentation'][0]) // 2

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

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))
    return list_iou, list_ciou



def main():
    import matplotlib.pyplot as plt

    gt_polygon_1 = shapely.geometry.Polygon(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
        ],
        # [[
        #     [0.1, 0.1],
        #     [0.9, 0.1],
        #     [0.9, 0.9],
        #     [0.1, 0.9]
        # ]]
    )
    # gt_polygon_2 = shapely.geometry.Polygon([
    #     [2, 2],
    #     [5, 0],
    #     [5, 6],
    #     [0, 4]
    # ])
    pred_polygon_1 = shapely.geometry.Polygon(
        [
            [0.1, 0.1],
            [10.1, 0],
            [9.9, 9],
            [9, 10.1],
            [0.1, 10]
        ],
        # [
        #     [0, 0],
        #     [10, 0],
        #     [10, 9],
        #     [10, 10],
        #     [9, 10],
        #     [0, 10]
        # ],
    )
    pred_polygons = [pred_polygon_1]
    gt_polygons = [gt_polygon_1]

    max_angle_diffs = compute_polygon_contour_measures(pred_polygons, gt_polygons, sampling_spacing=0.1, min_precision=0.5, max_stretch=2)
    # half_tangent_max_angles = compute_vertex_measures(pred_polygons, gt_polygons, min_precision=0.5)

    # print(cosine_similarities.mean())
    print(max_angle_diffs[0] * 180 / np.pi)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    plot_geometries(ax[0], gt_polygons)
    plot_geometries(ax[1], pred_polygons)
    # plot_geometries(ax[2], projected_pred_contours)
    for point in POINTS:
        ax[2].plot(*point.xy, marker="o", markersize=1)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
