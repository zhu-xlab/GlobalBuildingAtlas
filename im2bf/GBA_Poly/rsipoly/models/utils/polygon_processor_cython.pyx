from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pdb
from tqdm import tqdm
import numpy as np
import math
import shapely
import torch
import geopandas as gpd
from shapely.geometry import Polygon
import os
from pyproj import CRS
from pathlib import Path

class PolygonProcessor(ThreadPoolExecutor):

    def __init__(self, ring_sample_conf, out_dir, max_workers=8):
        super(PolygonProcessor, self).__init__(max_workers=max_workers)
        self.ring_sample_conf = ring_sample_conf
        self.out_dir = out_dir

    def sample_points_in_ring(self, ring, interval=None):

        interval = self.ring_sample_conf['interval'] if interval is None else interval

        try:
            ring_shape = shapely.LinearRing(ring)
        except ValueError:
            return None

        perimeter = ring_shape.length
        num_bins = max(round(perimeter / interval), 8)
        num_bins = max(num_bins, len(ring))

        bins = np.linspace(0, 1, num_bins)
        sampled_points = [ring_shape.interpolate(x, normalized=True) for x in bins]
        sampled_points = [[temp.x, temp.y] for temp in sampled_points]

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

        interval = self.ring_sample_conf['interval']
        length = self.ring_sample_conf['length']
        num_max_ring = self.ring_sample_conf['num_max_ring']
        ring_stride = self.ring_sample_conf['ring_stride']

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

            def format_results(polygons, out_dir='outputs/temp', img_meta=None, upscale=1):
                """Place holder to format result to dataset specific output."""
                filename = iter(img_meta['filename'])
                city_transform = img_meta['geo_transform']
                crs = img_meta['geo_crs']


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

                out_path = os.path.join(out_dir, '/'.join(filename.split('/')[-2:]))
                out_path = out_path.split('.')[0]
                temp = str(Path(out_path).parent)
                if not os.path.exists(temp):
                    os.makedirs(temp)

                gdf = gpd.GeoDataFrame(geometry=global_polygons)
                gdf.crs = crs
                gdf.to_file(out_path)

            length = self.ring_sample_conf['length']
            ring_stride = self.ring_sample_conf['ring_stride']

            pred_polygons = []
            for i in tqdm(range(len(all_ring_sizes)), desc=f'post processing on {img_meta["filename"]}...'):
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

            format_results(pred_polygons, out_dir=self.out_dir, img_meta=img_meta)

        self.submit(post_process_in_thread, pred_rings, ring_pred_next, all_idxes, all_ring_sizes, img_meta)
        # post_process_in_thread(pred_rings, ring_pred_next, all_idxes, all_ring_sizes, img_meta)


