import os
import rasterio
from rasterio.warp import transform_bounds
import pyproj
from pyproj import Transformer
from tqdm import tqdm

def calculate_area(tif_path):
    """
    Calculate the area covered by a geospatial .tif file using the updated pyproj API.

    Parameters:
    - tif_path: Path to the .tif file

    Returns:
    - Area in square meters.
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        left, bottom, right, top = transform_bounds(crs, 'EPSG:4326', *bounds)
        
        # Create a Transformer object for geographic to equal-area projection
        transformer_to_equal_area = Transformer.from_crs("EPSG:4326", 'EPSG:6933', always_xy=True)
        left_proj, bottom_proj = transformer_to_equal_area.transform(left, bottom)
        right_proj, top_proj = transformer_to_equal_area.transform(right, top)
        
        area = (right_proj - left_proj) * (top_proj - bottom_proj)
        return area


def calculate_total_area(folder_path):
    """
    Calculate the total area covered by all geospatial .tif files in a folder.

    Parameters:
    - folder_path: Path to the folder containing .tif files

    Returns:
    - Total area in square meters.
    """
    total_area = 0
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.tif'):
            tif_path = os.path.join(folder_path, filename)
            total_area += calculate_area(tif_path)
    
    return total_area

# Example usage
folder_path = '/home/Datasets/Dataset4EO/GlobalBF/so2sat/planet_global_processing/majority_voting_deflate'
# folder_path = '/home/Datasets/Dataset4EO/GlobalBF/so2sat/planet_global_processing/Continents/EUROPE/glcv103_guf_wsf'
total_area = calculate_total_area(folder_path)
print(f"The total area covered by the .tif files is approximately {total_area} square meters.")

