import os
import glob
import pdb
from planettools.utils import GeoJSONVisualizer
from tqdm import tqdm
import numpy as np

# tif_path = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/height_120m/oceania.vrt'
# tif_path = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/bf_120m/globe.vrt'
# tif_path = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/height_120m/north_america.vrt'
# tif_path = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/height_120m/south_america.vrt'
# tif_path = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/lod1_v5_120m/globe.vrt'
# out_dir = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/plots_lod1_v5'

tif_path = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/lod1_v5_volume_480m/globe.vrt'
out_dir = '/home/fahong/Datasets/ai4eo3/Global3D/rasters/plots_lod1_v5_volume'

# default_bins = [5.40632361e+02, 5.42705322e+03, 2.29277695e+04, 9.45178047e+04, 4.10035387e+05, 1.33344704e+08]
# default_bins = None
default_bins = [3.20971112e+02, 5.94804523e+03, 3.51230943e+04, 1.98466951e+05, 6.47502486e+05, 1e9]

bound_dict = {
    # 'Test1': [2724335, 3270022, 829426],
    # 'Africa': [-1711653, 5500663, 1240307]
    # 'Test2': [-8750832, -8474135, 2077084],
    # 'Oceania': [12544225, 17170226, -3164859],
    # 'North America': [-14418454,-7000369, 5036246],
    # 'South America': [-11055289, -3388255, -2716847]
    # 'Oceania': [12542283,-5525057, 17204302, -1298590]
    # 'Europe': [-1213899, 4416013, 6452622],
    # 'Africa': [-2619375, 6267170, 279571]
    # 'Asia': [9886692, 16105375, 4321120],
    # 'Munich': [1237152, 1334826, 6133139],
    # 'Cairo': [3186123, 3716314, 3549411],
    # 'Japan': [14349219, 16223015, 4685926],
    # 'Cairo': [3185441, 3718751, 3495905],
    # 'Moldova': [2976431, 3361945, 5900285],
    'Globe': [-14245046, -7399379.4493, 17371353, 11271636.5331],
    # 'Globe_web': [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244],
    # 'Globe_web': [-19865255, -7694453, 19945842, 11527190],
    # 'Tokyo': [15431368, 15659558, 4275523],
    # 'Cairo': [3410229, 3537817, 3509664]
    # 'Los Angles': [-13165315.2, -13162526.8, 4035419.9],
    # 'Sao Paulo': [-5405834, -5056537, -2676883],
    # 'Eastern China': [13227774, 13592021, 3658546],
    # 'Paris': [145190, 330915, 6254547],
    # 'San Fran': [-13646588, -13537369, 4543031],
    # 'Melbourne': [16052287, 16204327, -4547068],
    # 'Johannesburg': [3029091, 3198299, -3015494]
    # 'Shanghai': [13412330, 13589369, 3656671],
    # 'Tehran': [5527748, 5886252, 4251346],
    # 'Moscow': [4028984, 4352740, 7510754]
}
roi_list = [
    # 'Test1',
    # 'Africa',
    # 'Test2',
    # 'Oceania',
    # 'Asia',
    # 'Europe'
    # 'North America',
    # 'South America',
    # 'Munich'
    # 'Japan'
    # 'Moldova'
    'Globe'
    # 'Globe_web',
    # 'Cairo',
    # 'Los Angles',
    # 'Sao Paulo',
    # 'Eastern China',
    # 'Paris',
    # 'San Fran',
    # 'Melbourne',
    # 'Tokyo',
    # 'Johannesburg',
    # 'Shanghai',
    # 'Tehran',
    # 'Moscow'
]

# out_size = (2048, 2048)
out_size = (2048 * 2, 4096 * 2)
# out_size = (2048 * 4, 4096 * 4)
resize_type = 'resize_sparse'
# vis_conf = {'num_bins': 6, 'colormap': 'plasma'}
vis_conf = {
    'num_bins': 5, 'colormap': 'viridis',
    'colormap_colors': [[0,0,153], [96, 169, 23], [255, 255, 0], [255, 128, 0], [204, 0, 0]],
    # 'colormap_colors': [[0,0,153], [27, 161, 226], [255, 255, 0], [255, 128, 0], [204, 0, 0]],
    # 'resample_cfg': {'sample_type':'positive_sum'}
    'resample_cfg': {'sample_type':'max'}
    # 'resample_cfg': {'sample_type':'medium'}
}

os.makedirs(out_dir, exist_ok=True)
bin_edges = {}
visualizer = GeoJSONVisualizer(vis_conf)

for roi_name in tqdm(roi_list):
    bins = bin_edges.get(roi_name, default_bins)

    # if roi_name != 'WKY':
    #     continue
    bound = bound_dict[roi_name]
    if len(bound) == 3:
        bound = visualizer.get_square_roi(*bound)

    out_path = os.path.join(out_dir, f'{roi_name}.png')
    out_legend_path = os.path.join(out_dir, f'{roi_name}_legend.png')

    vis_img, cur_bins = visualizer.raster2png(
        tif_path, out_path, bound=bound, out_size=out_size,
        bins=bins, value_factor=1.0, resize_type=resize_type, min_height_value=-1,
        save_geotiff=True
    )
    if vis_img is None:
        print(f'roi {roi_name} is not found! Skip this roi')
        continue

    if bins is None:
        bins = cur_bins
        visualizer.save_colormap_legend(np.array(vis_conf['colormap_colors']), bins, out_legend_path)

