from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pdb
from tqdm import tqdm
import numpy as np
import math
import shapely
import torch
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import Polygon
import os
from pyproj import CRS
from pathlib import Path
import rsipoly.utils.polygon_utils_lydorn as polygon_utils

class PolygonProcessor(ThreadPoolExecutor):

    def __init__(self, ring_sample_conf, max_workers=16):
        super(PolygonProcessor, self).__init__(max_workers=max_workers)
        self.ring_sample_conf = ring_sample_conf

    def interpolate_ring_unequal_lengths(self, points, interval):
        """
        Interpolates points on a ring with unequal segment lengths using parameters ts.

        :param points: NumPy array of shape (N, 2) representing N 2-D points on the ring.
        :param ts: NumPy array of parameters for interpolation, where each element is in [0, 1].
        :return: NumPy array of shape (len(ts), 2) representing the interpolated points on the ring.
        """


        N = points.shape[0]  # Number of points
        
        # Calculate segment lengths
        segment_lengths = np.sqrt(((points - np.roll(points, -1, axis=0))**2).sum(axis=1))
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
        start_points = points[segment_indices]
        end_points = points[(segment_indices + 1) % N]  # Wrap around to the first point for the last segment
        interpolated_points = start_points + (end_points - start_points) * t_primes[:, np.newaxis]
        
        return interpolated_points

    def sample_points_in_ring(self, ring, interval=None):

        interval = self.ring_sample_conf['interval'] if interval is None else interval

        # try:
        #     ring_shape = shapely.LinearRing(ring)
        # except ValueError:
        #     return None

        # perimeter = ring_shape.length

        sampled_points = self.interpolate_ring_unequal_lengths(np.array(ring[:-1]), interval)

        # sampled_points = [ring_shape.interpolate(x, normalized=True) for x in bins]
        # sampled_points = [[temp.x, temp.y] for temp in sampled_points]

        return sampled_points

    def separate_ring(self, ring, crop_len, stride):

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


    def sample_rings(self, polygons):

        def split_list_into_parts(lst, k):
            n = len(lst)
            avg_length = n // k
            remainder = n % k

            parts = []
            start = 0
            for i in range(k):
                length = avg_length + (1 if i < remainder else 0)
                parts.append(lst[start:start+length])
                start += length

            return parts

        def sample_rings_in_thread(polygons, interval, length, ring_stride):

            all_rings = []
            all_ring_sizes = []
            all_idxes = []

            for i, rings in tqdm(enumerate(polygons), desc='sampling rings...'):
                sizes = []
                for j, ring in enumerate(rings):
                    sampled_ring = np.array(self.sample_points_in_ring(ring, interval=interval))
                    ring_parts = np.array(self.separate_ring(sampled_ring, crop_len=length, stride=ring_stride))
                    idx1 = np.array([i] * len(ring_parts))
                    idx2 = np.array([j] * len(ring_parts))
                    idx = np.stack([idx1, idx2], axis=1)
                    all_rings.append(ring_parts)
                    all_idxes.append(idx)
                    sizes.append(len(sampled_ring))

                all_ring_sizes.append(sizes)

            return all_rings, all_ring_sizes, all_idxes

        interval = self.ring_sample_conf['interval']
        length = self.ring_sample_conf['length']
        num_max_ring = self.ring_sample_conf['num_max_ring']
        ring_stride = self.ring_sample_conf['ring_stride']


        """
        # part_polygons = split_list_into_parts(polygons, 4)

        # futures = [sample_rings_in_thread(x, interval, length, ring_stride) for x in part_polygons]
        # futures = [self.submit(sample_rings_in_thread, x, interval, length, ring_stride) for x in part_polygons]

        all_rings = []
        all_ring_sizes = []
        all_idxes = []
        for future in as_completed(futures):
        # for future in futures:
            cur_rings, cur_ring_sizes, cur_idxes = future.result()
            # cur_rings, cur_ring_sizes, cur_idxes = future
            all_rings += cur_rings
            all_ring_sizes += cur_ring_sizes
            all_idxes += cur_idxes
        """

        all_rings, all_ring_sizes, all_idxes = sample_rings_in_thread(polygons, interval, length, ring_stride)

        all_rings = np.concatenate(all_rings, axis=0)
        all_idxes = np.concatenate(all_idxes, axis=0)
        all_rings = torch.tensor(all_rings)
        all_idxes = torch.tensor(all_idxes)

        return all_rings, all_idxes, all_ring_sizes


        for i, rings in tqdm(enumerate(polygons), desc='sampling rings...'):
            sizes = []
            for j, ring in enumerate(rings):
                sampled_ring = np.array(self.sample_points_in_ring(ring, interval=interval))
                ring_parts = np.array(self.separate_ring(sampled_ring, crop_len=length, stride=ring_stride))
                idx1 = np.array([i] * len(ring_parts))
                idx2 = np.array([j] * len(ring_parts))
                idx = np.stack([idx1, idx2], axis=1)
                all_rings.append(ring_parts)
                all_idxes.append(idx)
                sizes.append(len(sampled_ring))

            all_ring_sizes.append(sizes)

        all_rings = np.concatenate(all_rings, axis=0)
        all_idxes = np.concatenate(all_idxes, axis=0)
        all_rings = torch.tensor(all_rings)
        all_idxes = torch.tensor(all_idxes)

        return all_rings, all_idxes, all_ring_sizes

    def decode_ring_next(self, points, next_idxes, valid_mask=None, min_dis=2):
        x = 0
        pred_idxes = []
        while(next_idxes[x] > x and (valid_mask is None or valid_mask[x])):
            pred_idxes.append(x)
            x = next_idxes[x]

        if (valid_mask is None or valid_mask[x]) and (len(points) - x) >= min_dis:
            pred_idxes.append(x)

        pred_idxes = torch.tensor(pred_idxes).long()

        return pred_idxes


    def post_process(self, pred_rings, ring_pred_next, all_idxes, all_ring_sizes, img_meta):

        def post_process_in_thread(pred_rings, ring_pred_next, all_idxes, all_ring_sizes, img_meta):

            def format_results(polygons, img_meta=None, upscale=1):

                if len(polygons) == 0:
                    return None
                """Place holder to format result to dataset specific output."""
                filename = str(img_meta['filename'])
                city_transform = img_meta['geo_transform']
                crs = img_meta['geo_crs']

                in_root = str(img_meta['in_root'])
                out_root = str(img_meta['out_root'])

                rel_path = os.path.relpath(filename, in_root)
                out_path = os.path.join(out_root, rel_path)
                out_path = out_path.split('.tif')[0]

                if 'crop_boxes' in img_meta:
                    crop_boxes = [str(x) for x in img_meta['crop_boxes']]
                    out_path = os.path.join(out_path, '_'.join(crop_boxes))

                offset = np.array([0,0]).reshape(1,2)

                global_polygons = []

                for polygon in polygons:
                    new_rings = []
                    for ring in polygon:
                        new_ring = np.stack((city_transform * (ring * upscale + offset).permute(1,0).numpy()), axis=1)
                        if len(new_ring) >= 4:
                            new_rings.append(new_ring)

                    if len(new_rings) > 0:
                        new_polygon = Polygon(new_rings[0], new_rings[1:] if len(new_rings) > 1 else None)
                        global_polygons.append(new_polygon)

                # out_path = os.path.join(out_dir, '/'.join(filename.split('/')[-2:]))
                # out_path = out_path.split('.')[0]
                # temp = str(Path(out_path).parent)
                # if not os.path.exists(temp):
                #     os.makedirs(temp)

                gdf = gpd.GeoDataFrame(geometry=global_polygons)
                gdf.crs = crs
                gdf.to_file(out_path)

            length = self.ring_sample_conf['length']
            ring_stride = self.ring_sample_conf['ring_stride']

            pred_polygons = []
            file_str = str(img_meta["filename"]).split('/')[-1]
            box_str = '_'.join([str(x) for x in img_meta["crop_boxes"]])
            for i in tqdm(range(len(all_ring_sizes)), desc=f'post processing on {file_str}/{box_str}...'):
                cur_pred_polygon = []
                for j in range(len(all_ring_sizes[i])):
                    cur_mask = (all_idxes[:,0] == i) & (all_idxes[:,1] == j)
                    cur_rings = pred_rings[cur_mask]
                    cur_ring_len = all_ring_sizes[i][j]

                    cur_pred_ring = torch.zeros(cur_ring_len, 2)
                    cur_count = torch.zeros(cur_ring_len)

                    cur_ring_next = ring_pred_next[cur_mask]
                    ring_next = torch.zeros(cur_ring_len, cur_ring_len)

                    for k in range(cur_rings.shape[0]):
                        cur_valid_mask = (cur_rings[k] >= 0).all(dim=1)
                        temp = (torch.arange(length) + ring_stride * k) % cur_ring_len
                        cur_pred_ring[temp[cur_valid_mask]] += cur_rings[k][cur_valid_mask]
                        cur_count[temp] += 1
                        ring_next[temp[cur_valid_mask,None], temp[cur_valid_mask]] += cur_ring_next[k, cur_valid_mask][:, cur_valid_mask]

                    temp = cur_pred_ring[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)

                    pred_idxes = self.decode_ring_next(temp, ring_next.max(dim=1)[1])
                    cur_pred_polygon.append(temp[pred_idxes])

                pred_polygons.append(cur_pred_polygon)

            format_results(pred_polygons, img_meta=img_meta)

        self.submit(post_process_in_thread, pred_rings, ring_pred_next, all_idxes, all_ring_sizes, img_meta)
        # post_process_in_thread(pred_rings, ring_pred_next, all_idxes, all_ring_sizes, img_meta)

    def post_process_without_format(self, pred_rings, ring_pred_next, all_idxes, all_ring_sizes):

        def post_process_in_thread(pred_rings, ring_pred_next, all_idxes, all_ring_sizes):

            length = self.ring_sample_conf['length']
            ring_stride = self.ring_sample_conf['ring_stride']

            pred_polygons = []
            for i in range(len(all_ring_sizes)):
                cur_pred_polygon = []
                for j in range(len(all_ring_sizes[i])):
                    cur_mask = (all_idxes[:,0] == i) & (all_idxes[:,1] == j)
                    cur_rings = pred_rings[cur_mask]
                    cur_ring_len = all_ring_sizes[i][j]

                    cur_pred_ring = torch.zeros(cur_ring_len, 2)
                    cur_count = torch.zeros(cur_ring_len)

                    cur_ring_next = ring_pred_next[cur_mask]
                    ring_next = torch.zeros(cur_ring_len, cur_ring_len)

                    for k in range(cur_rings.shape[0]):
                        cur_valid_mask = (cur_rings[k] >= 0).all(dim=1)
                        temp = (torch.arange(length) + ring_stride * k) % cur_ring_len
                        cur_pred_ring[temp[cur_valid_mask]] += cur_rings[k][cur_valid_mask]
                        cur_count[temp] += 1
                        ring_next[temp[cur_valid_mask,None], temp[cur_valid_mask]] += cur_ring_next[k, cur_valid_mask][:, cur_valid_mask]

                    temp = cur_pred_ring[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)

                    pred_idxes = self.decode_ring_next(temp, ring_next.max(dim=1)[1])
                    cur_pred_polygon.append(temp[pred_idxes])

                pred_polygons.append(cur_pred_polygon)

            # format_results(pred_polygons, img_meta=img_meta)
            return pred_polygons
 

        pred_polygons = post_process_in_thread(pred_rings, ring_pred_next, all_idxes, all_ring_sizes)
        return pred_polygons

    def post_process_by_cls(self, pred_rings, ring_pred_cls, all_idxes, all_ring_sizes,
                            ring_pred_angle=None):

        def post_process_fun(pred_rings, ring_pred_cls, all_idxes, all_ring_sizes):

            def nms(probs, thre=0.5):
                A = torch.cat([probs[-1:], probs[:-1]], dim=0)
                B = torch.cat([probs[1:], probs[0:1]], dim=0)
                mask = (probs >= A) & (probs >= B)
                return ((probs > thre) & mask).nonzero().view(-1)


            length = self.ring_sample_conf['length']
            ring_stride = self.ring_sample_conf['ring_stride']
            cls_thre = self.ring_sample_conf.get('cls_thre', 0.5)

            pred_polygons = []
            for i in range(len(all_ring_sizes)): # iterate over batches
                cur_pred_polygon = []
                for j in range(len(all_ring_sizes[i])): # interate over polygons
                    cur_mask = (all_idxes[:,0] == i) & (all_idxes[:,1] == j)
                    cur_rings = pred_rings[cur_mask]
                    cur_rings_cls = ring_pred_cls[cur_mask]

                    cur_ring_len = all_ring_sizes[i][j]

                    cur_pred_ring = torch.zeros(cur_ring_len, 2)
                    cur_pred_ring_cls = torch.zeros(cur_ring_len, 2)
                    cur_count = torch.zeros(cur_ring_len)

                    for k in range(cur_rings.shape[0]): # interate over rings
                        cur_valid_mask = (cur_rings[k] >= 0).all(dim=1)
                        temp = (torch.arange(length) + ring_stride * k) % cur_ring_len
                        cur_pred_ring[temp[cur_valid_mask]] += cur_rings[k][cur_valid_mask]
                        cur_pred_ring_cls[temp[cur_valid_mask]] += cur_rings_cls[k][cur_valid_mask]

                        cur_count[temp] += 1

                    temp = cur_pred_ring[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)
                    temp2 = cur_pred_ring_cls[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)

                    probs = F.softmax(temp2, dim=-1)[:,1]
                    # pred_idxes = nms(probs, thre=cls_thre)
                    # cur_pred_polygon.append(temp[pred_idxes])

                    A = torch.cat([probs[-1:], probs[:-1]], dim=0)
                    B = torch.cat([probs[1:], probs[0:1]], dim=0)
                    mask = (probs >= A) & (probs >= B)

                    sort_v, sort_i = torch.sort(probs[mask], descending=True)
                    num_preds = max(4, (probs[mask] > cls_thre).sum())

                    pred_idxes = mask.nonzero().squeeze(1)[sort_i[:num_preds]].tolist()
                    cur_pred_polygon.append(temp[sorted(pred_idxes)])

                    # if ring_pred_angle is not None:
                    #     pdb.set_trace()

                pred_polygons.append(cur_pred_polygon)

            # format_results(pred_polygons, img_meta=img_meta)
            return pred_polygons
 

        pred_polygons = post_process_fun(pred_rings, ring_pred_cls, all_idxes, all_ring_sizes)
        return pred_polygons

    def post_process_by_clustering(self, pred_rings, pred_angles, pred_cnts, scores, all_idxes, all_ring_sizes):

        def nms(probs, thre=0.5):
            A = torch.cat([probs[-1:], probs[:-1]], dim=0)
            B = torch.cat([probs[1:], probs[0:1]], dim=0)
            mask = (probs >= A) & (probs >= B)
            return ((probs > thre) & mask).nonzero().view(-1)


        length = self.ring_sample_conf['length']
        ring_stride = self.ring_sample_conf['ring_stride']

        pred_polygons = []
        for i in range(len(all_ring_sizes)): # iterate over batches
            cur_pred_polygon = []
            for j in range(len(all_ring_sizes[i])): # interate over polygons
                cur_mask = (all_idxes[:,0] == i) & (all_idxes[:,1] == j)
                cur_rings = pred_rings[cur_mask]
                cur_ring_scores = scores[cur_mask]
                cur_ring_angles = pred_angles[cur_mask]
                cur_ring_cnts = pred_cnts[cur_mask]

                cur_ring_len = all_ring_sizes[i][j]
                cur_pred_ring = torch.zeros(cur_ring_len, 2)
                cur_count = torch.zeros(cur_ring_len)
                cur_cnts = torch.zeros(1)
                cur_scores = torch.zeros(cur_ring_len, cur_ring_len)
                cur_angles = torch.zeros(cur_ring_len)

                for k in range(cur_rings.shape[0]): # interate over rings
                    cur_valid_mask = (cur_rings[k] >= 0).all(dim=1)
                    temp = (torch.arange(length) + ring_stride * k) % cur_ring_len
                    cur_pred_ring[temp[cur_valid_mask]] += cur_rings[k][cur_valid_mask]
                    cur_scores[temp[cur_valid_mask,None], temp[cur_valid_mask]] += cur_ring_scores[k, cur_valid_mask][:, cur_valid_mask]
                    cur_angles[temp[cur_valid_mask]] += cur_ring_angles[k][cur_valid_mask]
                    cur_cnts += cur_ring_cnts[k]

                    # cur_pred_ring_cls[temp[cur_valid_mask]] += cur_rings_cls[k][cur_valid_mask]
                    cur_count[temp] += 1

                temp = cur_pred_ring[cur_count > 0] / cur_count[cur_count > 0].unsqueeze(1)
                cls_idx = polygon_utils.cluster_points_by_sim(cur_scores, eps=0.4)
                polygon = polygon_utils.cluster_to_polygons(temp, cur_angles, cls_idx)
                cur_pred_polygon.append(polygon)

                # if ring_pred_angle is not None:
                #     pdb.set_trace()

            pred_polygons.append(cur_pred_polygon)

        return pred_polygons
