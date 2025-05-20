from .builder import DATASETS
from .custom import EODataset
import pdb
import geopandas as gpd
import tifffile
import numpy as np
from shapely.geometry import Polygon
import os
from pyproj import CRS


@DATASETS.register_module()
class PolyBuildingDataset(EODataset):

    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, upscale=4, **kwargs):
        super(PolyBuildingDataset, self).__init__(**kwargs)
        self.transforms = self._dataset.geo_transforms
        self.global_polygons = {}
        self.upscale = upscale

    def format_results(self, results, imgfile_prefix, img_metas=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        assert len(img_metas) == 1
        filename = img_metas[0]['filename']
        city_name = filename.split('/')[-1].split('_')[0]
        city_transform = self.transforms[city_name]

        start_x, start_y = filename.split('/')[-1].split('.')[0].split('_')[-1].split('x')
        start_x, start_y = int(start_x), int(start_y)
        offset = np.array((start_x, start_y)).reshape(1, 2)

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

        return [[]]

    def dump_shp(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for city_name, polygons in self.global_polygons.items():
            out_path = os.path.join(out_dir, city_name + '.shp')
            gdf = gpd.GeoDataFrame(geometry=polygons)
            gdf.crs = CRS.from_epsg(32632)
            gdf.to_file(out_path)
