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
import rsipoly.utils.polygon_utils_lydorn as polygon_utils
import shapely


@DATASETS.register_module()
class PolyBuildingDatasetRasterV2(EODataset):

    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, upscale=4, pixel_width=0.75, pixel_height=0.75, thre=0.5, **kwargs):
        super(PolyBuildingDatasetRasterV2, self).__init__(**kwargs)
        # self.transforms = self._dataset.geo_transforms
        self.global_polygons = {}
        self.global_preds = {}
        self.upscale = upscale
        self.pixel_width=pixel_width
        self.pixel_height=pixel_height
        self.thre = thre

        percentile = 0.99
        num_bins = 25

    def format_results(self, results, imgfile_prefix, img_metas=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        # return [[]]

        for result in results:
            assert len(img_metas) == 1
            filename = img_metas[0]['filename']
            city_name = filename.split('/')[-1].split('_')[0]
            city_transform = img_metas[0]['geo_transform']
            crs = img_metas[0]['crs']

            # start_x, start_y = filename.split('/')[-1].split('.')[0].split('_')[-1].split('x')
            # start_x, start_y = int(start_x), int(start_y)
            # offset = np.array((start_x, start_y)).reshape(1, 2)

            if 'polygons' in result:
                polygons = result['polygons']

                if not city_name in self.global_polygons.keys():
                    self.global_polygons[city_name] = {}
                    self.global_polygons[city_name]['polygons'] = []

                self.global_polygons[city_name]['polygons'].extend(polygons)
                self.global_polygons[city_name]['transform'] = city_transform
                self.global_polygons[city_name]['crs'] = crs

                # for polygon in polygons:
                #     # polygon_utils.transform_polygon(polygon, city_transform)
                #     # for point in polygon:
                #     #     new_point = self.transforms[city_name] * point
                #     #     new_polygon.append(new_point)
                #     # new_polygon = Polygon(new_polygon)
                #     self.global_polygons[city_name][]append(polygon)

            elif 'polygons_v2' in result:
                if not city_name in self.global_polygons.keys():
                    self.global_polygons[city_name] = []

                for polygon in result['polygons_v2']:
                    new_rings = []
                    for ring in polygon:
                        new_ring = np.stack((city_transform * (ring * self.upscale + offset).permute(1,0).numpy()), axis=1)
                        if len(new_ring) >= 4:
                            new_rings.append(new_ring)

                    if len(new_rings) > 0:
                        new_polygon = Polygon(new_rings[0], new_rings[1:] if len(new_rings) > 1 else None)
                        self.global_polygons[city_name].append(new_polygon)

            if 'pred_mask' in result:
                if not city_name in self.global_preds.keys():
                    self.global_preds[city_name] = {}

                self.global_preds[city_name]['pred_mask'] = result['pred_mask']
                self.global_preds[city_name]['transform'] = city_transform
                self.global_preds[city_name]['crs'] = crs

        return [[]]

    def __len__(self):
        """Total number of samples of data."""
        # return self.geom_size
        return len(self.img_infos)

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
            geom_list = self.geom_dict[key],
            geom_probs = self.geom_probs[key],
            geom_bounds = self.geom_bounds[key]
        )
        return self.pipeline(results)

        num_sampled_polygons = 100
        sampled_idxes = np.random.choice(range(len(self.geom_lens)), size=num_sampled_polygons, p=self.geom_probs)
        results = dict(
            gt_features = [self.geoms[idx].__geo_interface__ for idx in sampled_idxes]
        )
        self.pre_pipeline(results)
        return self.pipeline(results)

        img_infos = self.img_infos[idx]
        gdal_ann = img_infos['gdal_info']['ann']
        gt_ann = img_infos['gt_info']['ann']
        img_info = img_infos['gdal_info']
        # img_info.pop('ann')

        results = dict(
            img_info=img_info,
            gdal_ann_info=gdal_ann,
            gt_ann_info=gt_ann
        )

        self.pre_pipeline(results)
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
            img_path = self.img_infos[idx]
        )

        self.pre_pipeline(results)
        return self.pipeline(results)


    def dump_shp(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for city_name, info in self.global_polygons.items():
            polygons = info['polygons']
            crs = info['crs']
            transform = info['transform']
            affine = np.array([transform.a, transform.b, transform.d, transform.e, transform.c, transform.f])
            polygons = [shapely.affinity.affine_transform(x, affine) for x in polygons]

            out_path = os.path.join(out_dir, city_name + '.shp')
            gdf = gpd.GeoDataFrame(geometry=polygons)
            gdf.crs = crs

            gdf.to_file(out_path)

        for city_name, info in self.global_preds.items():
            # crs_code = 'EPSG:32632'
            out_path = os.path.join(out_dir, city_name + '.tif')

            pred_mask = info['pred_mask']
            crs = info['crs']
            transform = info['transform']
            # affine = np.array([transform.a, transform.b, transform.d, transform.e, transform.c, transform.f])

            # probs = F.softmax(preds[0], dim=1)[0,1]
            # pdb.set_trace()
            # img = (probs > self.thre).cpu().numpy().astype(np.uint8) * 255
            # img = preds[0].max(dim=1)[1][0].cpu().numpy().astype(np.uint8) * 255

            # transform_list = np.array(self.transforms[city_name]).tolist()
            # top_left_x, top_left_y = transform_list[2], transform_list[5]
            # pixel_width, pixel_height = self.pixel_width, self.pixel_height
            # transform = from_origin(top_left_x, top_left_y, pixel_width, pixel_height)

            """
            tifffile.imwrite(
                out_path, img, photometric='minisblack',
                metadata={'transform': transform_list, 'crs': crs_code}
            )
            """

            with rasterio.open(
                out_path, 'w', driver='GTiff', height=pred_mask.shape[0], width=pred_mask.shape[1], count=1,
                dtype=str(pred_mask.dtype), crs=crs, transform=transform
            ) as dst:
                dst.write(pred_mask, 1)
