from .builder import DATASETS
from .custom import EODataset
import pdb
import geopandas as gpd
import tifffile
import numpy as np
from shapely.geometry import Polygon
import os
from pyproj import CRS
import rasterio
from rasterio.transform import from_origin
import torch.nn.functional as F


@DATASETS.register_module()
class PolyBuildingDatasetShape(EODataset):

    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, upscale=4, pixel_width=0.75, pixel_height=0.75, thre=0.5, has_shape=True, **kwargs):
        super(PolyBuildingDatasetShape, self).__init__(**kwargs)
        # self.transforms = self._dataset.geo_transforms
        self.global_polygons = {}
        self.global_preds = {}
        self.upscale = upscale
        self.pixel_width=pixel_width
        self.pixel_height=pixel_height
        self.thre = thre
        self.has_shape = has_shape

        if has_shape:
            self._init_geom()

    def _init_geom(self):
        percentile = 0.99
        num_bins = 25

        geom_dict = self._dataset.geom_dict
        geom_json_dict = {}
        geom_probs = {}
        geom_lens = {}
        geom_bounds = {}
        geom_size = 0

        for key, geoms in geom_dict.items():
            print(f'processing annotations: {key}, sizes: {len(geoms)}')
            lens = np.array([x.length for x in geoms])
            sorted_lens = np.sort(lens)
            l, r = 0, sorted_lens[round(len(lens) * percentile)]
            bins = np.linspace(l, r, num_bins).tolist()
            bins.append(sorted_lens[-1] + 1e-6)
            hist_label = np.digitize(lens, bins)
            hist_freq, _ = np.histogram(lens, bins)

            p = 1 / hist_freq[hist_label - 1]
            p = p / p.sum()

            geom_lens[key] = lens
            geom_probs[key] = p
            geom_size += len(lens)

            bounds = np.array([g.bounds for g in geoms])
            geom_bounds[key] = bounds

            geom_json_dict[key] = [g.__geo_interface__ for g in geoms]

        self.geom_dict = geom_dict
        self.geom_json_dict = geom_json_dict
        self.geom_lens = geom_lens
        self.geom_probs = geom_probs
        self.geom_size = geom_size
        self.geom_bounds = geom_bounds

    def format_results(self, results, imgfile_prefix, img_metas=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""

        return [[]]
        assert len(img_metas) == 1
        filename = img_metas[0]['filename']
        city_name = str(filename).split('/')[-1].split('_')[0]
        city_transform = self.transforms[city_name]

        # start_x, start_y = filename.split('/')[-1].split('.')[0].split('_')[-1].split('x')
        # start_x, start_y = int(start_x), int(start_y)
        # offset = np.array((start_x, start_y)).reshape(1, 2)

        offset = np.array([0,0]).reshape(1,2)

        if 'polygons' in results:
            polygons = results['polygons'][0]
            offset_polygons = [(polygon / self.upscale + offset) for polygon in polygons]

            if not city_name in self.global_polygons.keys():
                self.global_polygons[city_name] = []

            for polygon in offset_polygons:
                new_polygon = []
                for point in polygon:
                    new_point = self.transforms[city_name] * point
                    new_polygon.append(new_point)
                new_polygon = Polygon(new_polygon)
                self.global_polygons[city_name].append(new_polygon)

        elif 'polygons_v2' in results:
            if not city_name in self.global_polygons.keys():
                self.global_polygons[city_name] = []

            for polygon in results['polygons_v2']:
                new_rings = []
                for ring in polygon:
                    new_ring = np.stack((city_transform * (ring * self.upscale + offset).permute(1,0).numpy()), axis=1)
                    if len(new_ring) >= 4:
                        new_rings.append(new_ring)

                if len(new_rings) > 0:
                    new_polygon = Polygon(new_rings[0], new_rings[1:] if len(new_rings) > 1 else None)
                    self.global_polygons[city_name].append(new_polygon)

        if 'seg_logits' in results:
            if not city_name in self.global_preds.keys():
                self.global_preds[city_name] = []
            self.global_preds[city_name].append(results['seg_logits'])

        return [[]]

    def __len__(self):
        """Total number of samples of data."""
        # return len(self.img_infos)
        return 10000 if self.has_shape else len(self._dataset)
        # return len(self._dataset)
        # return self.geom_size if self.has_shape else 
        # return 100000

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        key = np.random.choice(list(self.geom_dict.keys()))
        results = dict(
            geom_list_json = self.geom_json_dict[key],
            geom_list = self.geom_dict[key],
            geom_probs = self.geom_probs[key],
            geom_bounds = self.geom_bounds[key]
        )
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        results = dict(
            img_info=dict(
                filename=self.img_infos[idx],
            ),
            in_root=self._dataset.root,
            in_base_dir=self._dataset.in_base_dir,
            out_root=self._dataset.out_root
        )

        self.pre_pipeline(results)
        return self.pipeline(results)


    def dump_shp(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for city_name, polygons in self.global_polygons.items():
            out_path = os.path.join(out_dir, city_name + '.shp')
            gdf = gpd.GeoDataFrame(geometry=polygons)
            gdf.crs = CRS.from_epsg(32632)
            gdf.to_file(out_path)

        for city_name, preds in self.global_preds.items():
            crs_code = 'EPSG:32632'
            out_path = os.path.join(out_dir, city_name.split('.')[0] + '.tif')

            probs = F.softmax(preds[0], dim=1)[0,1]
            # pdb.set_trace()
            img = (probs > self.thre).cpu().numpy().astype(np.uint8) * 255

            # img = preds[0].max(dim=1)[1][0].cpu().numpy().astype(np.uint8) * 255
            transform_list = np.array(self.transforms[city_name]).tolist()
            top_left_x, top_left_y = transform_list[2], transform_list[5]
            pixel_width, pixel_height = self.pixel_width, self.pixel_height
            transform = from_origin(top_left_x, top_left_y, pixel_width, pixel_height)

            """
            tifffile.imwrite(
                out_path, img, photometric='minisblack',
                metadata={'transform': transform_list, 'crs': crs_code}
            )
            """

            with rasterio.open(
                out_path, 'w', driver='GTiff', height=img.shape[0], width=img.shape[1], count=1,
                dtype=str(img.dtype), crs=crs_code, transform=transform
            ) as dst:
                dst.write(img, 1)
