import os
import glob
import pdb
from planettools.utils import GeoJSONVisualizer
from tqdm import tqdm
import numpy as np

tif_pattern_dict = {
    'Reference': '/home/fahong/Datasets/ai4eo3/Global3D/inference/figures/gt/rasters2/{}.tif',
    'Ours': '/home/fahong/Datasets/ai4eo3/Global3D/inference/figures/ours/rasters2/{}.tif',
    'Ours_height': '/home/fahong/Datasets/ai4eo3/Global3D/inference_v2/raster_28cities/mosaicked/{}.tif',
    'Google25D': '/home/fahong/Datasets/ai4eo3/Global3D/inference/google25d/ndsm/{}.tif',
    '3D-GloBFP': '/home/fahong/Datasets/ai4eo3/Global3D/inference/figures/3dglobfp/rasters/{}.tif',
    'Microsoft': '/home/fahong/Datasets/ai4eo3/Global3D/inference/figures/microsoft/rasters/{}.tif',
    'WSF3D': '/home/fahong/Datasets/ai4eo3/Global3D/inference_v2/wsf3d/height/{}.tif',
    'GHSL3D': '/home/fahong/Datasets/ai4eo3/Global3D/inference/ghsl3d/{}.tif',
}
# resize_type = 'scipy'
resize_type = 'resize_sparse'
resize_order = 0

bound_dict = {
    # 'WKY': [15046039.6, 15049358.2, 4060486.4],
    # 'BUE': [-6509884.1, -6505212.9, -4106014.3],
    # 'LCT': [16376489.2, 16383018.9, -5077323.2],
    # 'LCT_zoom': [16378775.50, 16379609.34, -5077475.56],
    # 'LCT_zoom': [16379098.23, 16379922.62, -5077115.70]
    # 'OMH': [-10684567.1, -10678931.9, 5051569.8],
    # 'FBG': [869030.2, 875327.5, 6108058.1],
    # 'MDN': [-8415975.5, -8409509.3, 697057.2],
    # 'MDN': [-8417404.0, -8413441.4, 693089.4],
    # 'MDN_zoom': [-8416634.97, -8415970.65, 693912.70],
    # 'BUE': [-6507943.4, -6507400.9, -4107057.9]
    # 'PTL': [-13657998, -13647276, 5703872],
    # 'PTL_zoom': [-13657240.17, -13656327.37, 5703437.40],
    # 'BDX': [-70635.1, -65105.6, 5597810.7],
    # 'BDX_zoom': [-65814.9, -65138.1, 5596137.5],
    # 'WKY': [15046639.2, 15050253.7, 4059555.2]
    # 'FKK': [14513178.3, 14519378.1, 3971146.8]
    # 'FKK': [14514196.6, 14520238.8, 3973543.4],
    'WKY': [15046386.5, 15051127.9, 4059348.0],
    'WKY_zoom': [15047046.59, 15047729.28, 4059824.43]
    # 'FKK': [14510375.3, 14514617.1, 3972841.2]
}

out_size = (256, 256)
# vis_conf = {'num_bins': 6, 'colormap': 'plasma'}
vis_conf = {
    'num_bins': 5, 'colormap': 'jet',
    # 'colormap_colors': [[0,0,153], [27, 161, 226], [96, 169, 23], [255, 255, 0], [255, 128, 0], [204, 0, 0]]
    'colormap_colors': [[0,0,153], [96, 169, 23], [255, 255, 0], [255, 128, 0], [204, 0, 0]]
}

roi_pattern = '/home/fahong/Datasets/ai4eo3/Global3D/inference/roi/*.geojson'
out_dir = '/home/fahong/Datasets/ai4eo3/Global3D/inference/figures/plots_zoom'
os.makedirs(out_dir, exist_ok=True)
# roi_list = ['WKY', 'BUE', 'OMH', 'FBG', 'LCT']
roi_list = ['WKY']

visualizer = GeoJSONVisualizer(vis_conf)

roi_paths = glob.glob(roi_pattern)
for roi_path in tqdm(roi_paths):
    roi_name = roi_path.split('/')[-1].split('.')[0]
    # bins = bin_edges.get(roi_name, None)
    bins = None

    if not roi_name in roi_list:
        continue

    # if roi_name != 'WKY':
    #     continue

    if roi_name in bound_dict:
        bound = bound_dict[roi_name]
        bound = visualizer.get_square_roi(*bound)
    else:
        bound = None

    for i, (product, tif_pattern) in enumerate(tif_pattern_dict.items()):
        tif_path = tif_pattern.format(roi_name)

        if os.path.exists(tif_path):
            out_path = os.path.join(out_dir, f'{product}_{roi_name}.png')
            out_legend_path = os.path.join(out_dir, f'{roi_name}_legend.png')
            value_factor = 0.1 if product == 'WSF3D' else 1.0
            _, cur_bins = visualizer.raster2png(
                tif_path, out_path, bound=bound, out_size=out_size,
                bins=bins, value_factor=value_factor, resize_type=resize_type,
                resize_order=resize_order
            )

            if bins is None:
                bins = cur_bins
                visualizer.save_colormap_legend(np.array(vis_conf['colormap_colors']), bins, out_legend_path)

            if roi_name + '_zoom' in bound_dict:
                zoom_bound = bound_dict[roi_name + '_zoom']
                zoom_bound = visualizer.get_square_roi(*zoom_bound)

                zoom_out_path = os.path.join(out_dir, f'{product}_{roi_name}_zoom.png')
                _, cur_bins = visualizer.raster2png(
                    tif_path, zoom_out_path, bound=zoom_bound, out_size=out_size,
                    bins=bins, value_factor=value_factor, resize_type=resize_type,
                    resize_order=resize_order
                )
