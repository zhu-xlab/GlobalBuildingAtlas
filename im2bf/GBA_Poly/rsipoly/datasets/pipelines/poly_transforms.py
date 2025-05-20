import mmcv
import cv2
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
import pdb
import math
import shapely
from skimage import measure

from ..builder import PIPELINES


@PIPELINES.register_module()
class CreatePolyAnn(object):

    def __init__(self, num_max_vertices=1024, add_contour=False):
        self.num_max_vertices = num_max_vertices
        self.add_contour = add_contour

    def __call__(self, results):

        H, W, _ = results['img'].shape
        # polygons = results['ann_info']['masks']
        polygons = results['polygons']

        poly_sizes = [len(polygon) for polygon in polygons]
        # vertices_list = [vertex for polygon in polygons for vertex in polygon]
        if len(polygons) > 0:
            vertices_list = np.concatenate(polygons, axis=0)
        else:
            vertices_list = np.zeros((0,2))

        num_vertices = len(vertices_list)

        # vertices = np.zeros((self.num_max_vertices, 2)).astype(np.int)
        vertices = np.zeros((self.num_max_vertices, 2))
        permute_mat = np.zeros((self.num_max_vertices, self.num_max_vertices))
        mask = np.zeros((1, H, W)).astype(np.uint8)

        if num_vertices > 0:
            vertices_list[:, 0] = vertices_list[:, 0].clip(0, W-1)
            vertices_list[:, 1] = vertices_list[:, 1].clip(0, H-1)
            mask[0, vertices_list[:, 1].round().astype(np.int),
                 vertices_list[:, 0].round().astype(np.int)] = 1

            vertices[:num_vertices, 0] = vertices_list[:self.num_max_vertices, 0]
            vertices[:num_vertices, 1] = vertices_list[:self.num_max_vertices, 1]

        vertices_mask = np.zeros(self.num_max_vertices).astype(np.uint8)
        vertices_mask[:num_vertices] = 1

        if self.add_contour and num_vertices < self.num_max_vertices:
            img = results['img'][:, :, 0]
            contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            num_add_points = np.random.randint(0, min(len(contours), self.num_max_vertices - num_vertices)+1)
            if num_add_points > 0:
                contours = np.concatenate(contours, axis=0).reshape(-1, 2)
                rand_idx = np.random.permutation(len(contours))
                vertices[num_vertices:num_vertices+num_add_points] = contours[rand_idx[:num_add_points], :]
                vertices_mask[num_vertices:num_vertices+num_add_points] = 1
                permute_mat[num_vertices:num_vertices+num_add_points, num_vertices:num_vertices+num_add_points] = 1

        # last_idx = 0
        # for poly_size in poly_sizes:
        #     permute_mat[last_idx:last_idx+poly_size, last_idx:last_idx+poly_size] = 1
        #     last_idx += poly_size
        start_ver_idx = 0
        ver_idx = 0
        for polygon in polygons:
            if start_ver_idx + len(polygon) >= self.num_max_vertices:
                break
            for _ in polygon[:-1]:
                permute_mat[ver_idx, ver_idx+1] = 1
                ver_idx += 1
            permute_mat[ver_idx, start_ver_idx] = 1
            ver_idx += 1
            start_ver_idx += len(polygon)


        results['vertices'] = vertices
        results['vertices_mask'] = vertices_mask
        results['permute_matrix'] = permute_mat
        results['mask'] = mask

        return results

@PIPELINES.register_module()
class LoadPolyAnn(object):

    def __init__(self, num_bins=36, add_reverse=False):
        self.num_bins = num_bins
        self.add_reverse = add_reverse

    def __call__(self, results):

        # polygons = results['ann_info']['masks']
        polygons = results['img_info']['ann']['masks']
        polygons = [np.array(polygon) for polygon in polygons]
        results['polygons'] = polygons

        degs = []
        for polygon in polygons:
            ext_polygon = np.concatenate([polygon, polygon[0:1]])
            diff = ext_polygon[1::1] - ext_polygon[:-1]
            deg = np.mod(np.arctan2(- diff[:,1], diff[:,0]), 2 * np.pi)
            degs.append(deg)

        if self.add_reverse:
            rev_degs = []
            for polygon in polygons:
                ext_polygon = np.concatenate([polygon[-1:], polygon])
                diff =  ext_polygon[:-1] - ext_polygon[1:]
                rev_deg = np.mod(np.arctan2(- diff[:,1], diff[:,0]), 2 * np.pi)
                rev_degs.append(rev_deg)

        if len(degs) > 0:
            degs = np.concatenate(degs, axis=0)
            num_bins = self.num_bins if not self.add_reverse else self.num_bins // 2

            bins = np.linspace(0, 2 * math.pi, num_bins + 1)
            bin_indices = np.digitize(degs, bins, right=True)  # right closed
            degs = np.eye(num_bins)[bin_indices - 1]

            if self.add_reverse:
                rev_degs = np.concatenate(rev_degs, axis=0)
                assert self.num_bins % 2 == 0
                rev_bins = np.linspace(0, 2 * math.pi, num_bins + 1)
                rev_bin_indices = np.digitize(rev_degs, rev_bins, right=True)  # right closed
                rev_degs = np.eye(num_bins)[rev_bin_indices - 1]
                degs = np.concatenate([degs, rev_degs], axis=-1)
        else:
            degs = np.zeros((0, self.num_bins))

        results['gt_degrees'] = degs

        return results

@PIPELINES.register_module()
class LoadPolyAnnV2(object):

    def __init__(self, num_bins=36, add_reverse=False):
        self.num_bins = num_bins
        self.add_reverse = add_reverse

    def __call__(self, results):

        # polygons = results['ann_info']['masks']

        # polygons = results['img_info']['ann']['masks']
        # polygons = [np.array(polygon) for polygon in polygons]
        # results['polygons'] = polygons
        gdal_features = results['gdal_ann_info']
        gt_features = results['gt_ann_info']

        results['gdal_features'] = gdal_features
        results['gt_features'] = gt_features

        return results

@PIPELINES.register_module()
class CreateGTPointsFromPolygons(object):

    def __call__(self, results):

        H, W, _ = results['img'].shape
        polygons = results['polygons']
        sizes = [len(polygon) for polygon in polygons]

        if len(polygons) > 0:
            vertices_list = np.concatenate(polygons, axis=0)
            # gt_labels = np.array([[idx+1] * size for idx, size in enumerate(sizes)])
            gt_labels = [np.array([idx+1] * size) for idx, size in enumerate(sizes)]
            gt_labels = np.concatenate(gt_labels, axis=0)
        else:
            vertices_list = np.zeros((0,2))
            gt_labels = np.zeros((0,))

        results['gt_points'] = vertices_list
        results['gt_labels'] = gt_labels

        mask = np.zeros((1, H, W)).astype(np.uint8)

        if len(polygons) > 0:
            temp1 = vertices_list[:, 0].clip(0, W-1)
            temp2 = vertices_list[:, 1].clip(0, H-1)
            mask[0, temp2.round().astype(np.int),
                 temp1.round().astype(np.int)] = 1

        results['mask'] = mask

        return results

@PIPELINES.register_module()
class CreateContours(object):

    def __call__(self, results):

        img = results['img']
        H, W, _ = img.shape
        mask = (img[:, :, 0] > 0).astype(np.uint8)
        # contours = self.get_corners(mask)
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # contours = measure.find_contours(mask, 127)
        # sizes = [len(contour) for contour in contours]
        # contour_labels = np.concatenate([np.array([idx]*size, dtype=np.int32) for idx, size in enumerate(sizes)])

        if len(contours) > 0:
            contours = np.concatenate(contours, axis=0).reshape(-1, 2)
        else:
            contours = np.zeros((0,2), dtype=np.int32)

        # offset = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]])
        # contours.reshape(-1, 1, 2)

        num_labels, labeled_image = cv2.connectedComponents(mask)
        contour_labels = labeled_image[contours[:,1], contours[:,0]]
        # contour_labels = np.zeros(len(contours), dtype=np.int32)

        # Iterate through the detected contours
        # corner_points = []

        # for contour in contours:
        #     corner_points.append(contour.reshape(-1, 2))
            # Approximate the contour to reduce the number of points
            # epsilon = 0.03 * cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, epsilon, True)

            # # Extract the corner points of the contour
            # for point in approx:
            #     x, y = point[0]
            #     corner_points.append((x, y))

        results['contours'] = contours.astype(np.float32)
        results['contour_labels'] = contour_labels
        results['comp_mask'] = labeled_image

        return results

        # corner_points now contains the corner points of the binary mask

@PIPELINES.register_module()
class ParseShape(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self, sample_type='random', num_sampled_polygons=100):
        self.sample_type = sample_type
        self.num_sampled_polygons = num_sampled_polygons

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        geom_list = results['geom_list_json']
        geom_probs = results['geom_probs']

        if self.sample_type == 'random':
            sampled_idxes = np.random.choice(range(len(geom_list)), size=self.num_sampled_polygons, p=geom_probs)
            # gt_features = [geom_list[idx].__geo_interface__ for idx in sampled_idxes]
            gt_features = [geom_list[idx] for idx in sampled_idxes]
            results['gt_features'] = gt_features

        elif self.sample_type == 'all':
            results['gt_features'] = geom_list
            # results['gt_features'] = [g.__geo_interface__ for g in geom_list]


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class ErodeGT(object):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, results):

        new_seg_fields = []
        for key in results['seg_fields']:
            mask = results[key]
            unique_labels = np.unique(mask)

            if 0 in unique_labels:
                unique_labels = unique_labels[1:]  # Remove background label if present

            # Define the structuring element (kernel) for erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))

            # Placeholder for the eroded mask
            eroded_mask = np.zeros_like(mask)

            # Erode each component individually
            for label in unique_labels:
                component_mask = (mask == label).astype(np.uint8)  # Create binary mask for current component
                eroded_component = cv2.erode(component_mask, kernel, iterations=1)

                # Place the eroded component back into the eroded_mask
                eroded_mask[eroded_component == 1] = label

            results['eroded_' + key] = eroded_mask
            new_seg_fields.append('eroded_' + key)

        results['seg_fields'].extend(new_seg_fields)

        return results

@PIPELINES.register_module()
class CropFeaturesToBounds(object):
    def __init__(self):
        pass

    def shapely2json(self, geom):
        results = []
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                results.extend(self.shapely2json(poly))
        elif geom.geom_type == 'Polygon':
            results.append(shapely.geometry.mapping(geom))
        elif geom.geom_type == 'GeomtryCollection':
            for poly in geom.geoms:
                results.extend(self.shapely2json(poly))
        else:
            pass

        return results


    def __call__(self, results):

        if 'features' in results:
            H, W, _ = results['img'].shape
            bound_shape = shapely.Polygon([(0, 0), (W, 0), (W, H), (0, H)])

            new_features = []
            for feature in results['features']:
                cur_shape = shapely.geometry.shape(feature)
                if cur_shape.is_valid:
                    intersect = cur_shape.intersection(bound_shape)
                    if not intersect.is_empty:
                        new_features.extend(self.shapely2json(intersect))
                else:
                    pass
                    # print(f'An invalid geometry found in GT: {feature}, ignore it')


            results['features'] = new_features

        return results


@PIPELINES.register_module()
class LoadShape(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(
        self, sample_type='random', num_sampled_polygons=100,
        to_pixel_coords=True, gsd_range=[0.3, 0.3], normalize=True,
        min_perimeter=None
    ):
        self.sample_type = sample_type
        self.num_sampled_polygons = num_sampled_polygons
        self.to_pixel_coords=to_pixel_coords
        self.gsd_range=gsd_range
        self.normalize=normalize
        self.min_perimeter=min_perimeter
        assert len(gsd_range) == 2

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        geom_list = results['geom_list_json']
        geom_probs = results['geom_probs']
        random_gsd = np.random.rand() * (self.gsd_range[1] - self.gsd_range[0]) + self.gsd_range[0]

        if self.sample_type == 'random':
            sampled_idxes = np.random.choice(range(len(geom_list)), size=self.num_sampled_polygons, p=geom_probs)
            # gt_features = [geom_list[idx].__geo_interface__ for idx in sampled_idxes]
            gt_features = [geom_list[idx] for idx in sampled_idxes]
            results['features'] = gt_features

        elif self.sample_type == 'all':
            results['features'] = geom_list
            # results['gt_features'] = [g.__geo_interface__ for g in geom_list]

        if self.to_pixel_coords:
            new_features = []
            for feature in results['features']:
                new_feature = feature.copy()
                new_rings = []
                offset = np.array(feature['coordinates'][0]).min(axis=0)
                for ring in feature['coordinates']:
                    new_ring = (np.array(ring) / random_gsd)
                    if self.normalize:
                        new_ring -= offset / random_gsd
                    new_rings.append(new_ring.tolist())

                if self.min_perimeter is not None:
                    ext = np.array(new_rings[0])
                    perimeter = (((ext[1:] - ext[:-1]) ** 2).sum(axis=-1) ** 0.5).sum()
                    if perimeter < self.min_perimeter:
                        new_rings = [(np.array(x) * self.min_perimeter / perimeter).tolist() for x in new_rings]

                new_feature['coordinates'] = new_rings
                new_features.append(new_feature)

            results['features'] = new_features


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomCropShapeV2(object):

    def __init__(self, crop_size=(1024, 1024), gsd=0.75):
        self.crop_size = crop_size
        self.gsd = gsd

    def get_crop_bbox(self, H, W, h, w):
        """Randomly get a crop bounding box."""
        margin_h = max(H - h, 0)
        margin_w = max(W - w, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + h
        crop_x1, crop_x2 = offset_w, offset_w + w

        return crop_y1, crop_y2, crop_x1, crop_x2


    def __call__(self, results):
        gt_features = results['features']
        bounds = results['geom_bounds']

        pixel_bounds = bounds / self.gsd

        start_offset = pixel_bounds.min(axis=0)[[0,1]].round().astype(np.int)
        img_width = (pixel_bounds[:,2].max() - pixel_bounds[:,0].min()).round().astype(np.int)
        img_height = (pixel_bounds[:,3].max() - pixel_bounds[:,1].min()).round().astype(np.int)
        offset = np.concatenate([start_offset, start_offset], axis=0)
        off_bounds = pixel_bounds - offset

        while True:
            crop_bbox = self.get_crop_bbox(img_height, img_width, self.crop_size[0], self.crop_size[1])
            start_y, end_y, start_x, end_x = crop_bbox

            valid_idxes = (off_bounds[:,0] > start_x) & (off_bounds[:,1] > start_y) & (off_bounds[:,2] < end_x) & (off_bounds[:,3] < end_y)
            if valid_idxes.sum() > 0:
                break

        pdb.set_trace()
        # raster_offset = bounds.min(axis=0)[[0,1]]
        bound_gt_features = [gt_features[x] for x in valid_idxes.nonzero()[0]]
        for g in bound_gt_features:
            rings = g['coordinates']
            new_rings = []
            for ring in rings:
                coords = np.array(ring) / self.gsd
                off_coords = coords - start_offset - np.array([start_x, start_y])
                new_rings.append(off_coords)

            g['coordinates'] = new_rings

        results['gt_features'] = bound_gt_features

        return results


