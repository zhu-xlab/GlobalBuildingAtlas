import math
import numpy as np
import geopandas as gpd
import rasterio
import pyproj
from rasterio.transform import from_bounds
from PIL import Image
from matplotlib import cm
from shapely.geometry import Polygon
import scipy
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class GeoJSONVisualizer:
    def __init__(self, vis_conf: dict = {}):
        # Set default configurations
        self.height_key = vis_conf.get('height_key', 'Height')
        self.out_epsg   = vis_conf.get('out_epsg', 3857)
        self.num_bins   = vis_conf.get('num_bins', 6)
        self.colormap   = vis_conf.get('colormap', 'viridis')
        self.colormap_colors   = vis_conf.get('colormap_colors', None)
        self.vis_conf = vis_conf

    def rasterize_to_tiff(self, geojson_path, roi_path, output_path, gsd: float, no_height_value:
                          float=-1, merge_alg='max'):
        pdb.set_trace() # merge_alg
        # 1. Load features and ROI as GeoDataFrames
        gdf = gpd.read_file(geojson_path)
        roi_gdf = gpd.read_file(roi_path)
        # 2. Reproject to the desired output CRS (if not already in it)
        gdf = gdf.to_crs(epsg=self.out_epsg)
        roi_gdf = roi_gdf.to_crs(epsg=self.out_epsg)
        # 3. Ensure height_key exists and fill missing heights with -1
        if self.height_key not in gdf.columns:
            gdf[self.height_key] = no_height_value
        else:
            gdf[self.height_key] = gdf[self.height_key].fillna(-1)

        gdf = gdf[gdf.is_valid]

        # 4. Clip features to ROI geometry (to restrict to the area of interest)
        roi_geometry = roi_gdf.geometry.unary_union  # combine ROI geometries into one
        gdf_clipped = gpd.clip(gdf, roi_geometry)
        # 5. Determine output raster bounds and resolution
        minx, miny, maxx, maxy = roi_gdf.total_bounds  # ROI bounding box in out_epsg
        width  = math.ceil((maxx - minx) / gsd)  # number of pixels in x-direction
        height = math.ceil((maxy - miny) / gsd)  # number of pixels in y-direction
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        # 6. Prepare shapes (geometry, value) for rasterization
        shapes = list((geom, value) for geom, value in zip(gdf_clipped.geometry, gdf_clipped[self.height_key]))
        # Rasterize to array with background fill = -2 (float32 array)
        raster_arr = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=-1.0,
            transform=transform,
            dtype='float32',
            all_touched=True,
            merge_alg=merge_alg
        )
        # 7. Write the raster array to a GeoTIFF file
        out_profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': f'EPSG:{self.out_epsg}',
            'transform': transform
        }
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            dst.write(raster_arr, 1)

    @staticmethod
    def get_histogram_equalized_bins(positive_vals, num_bins=6):
        """
        Compute histogram equalized bin edges for a given set of positive height values.
        
        The bins are determined such that each bin contains approximately the same number of values.

        Parameters
        ----------
        positive_vals : np.ndarray
            Array of positive height values.
        num_bins : int, optional
            The number of bins to create (default is 6).

        Returns
        -------
        bin_edges : np.ndarray
            The computed bin edges.
        """
        if len(positive_vals) == 0:
            return np.array([0, 1])  # Return a dummy bin if there are no positive values

        # Sort the values
        sorted_vals = np.sort(positive_vals)

        # Compute the quantiles at uniform intervals
        # percentiles = np.linspace(0, 100, num_bins + 1)  # Example: 0%, 20%, 40%, ..., 100%

        alpha = 2.5
        x = np.linspace(0, 1, num_bins+1)  # Uniform spacing
        percentiles = 30 + (100 - 30) * (1 - (1 - x) ** alpha)

        bin_edges = np.percentile(sorted_vals, percentiles)

        # Ensure bin edges are strictly increasing (fix duplicate values)
        bin_edges = np.unique(bin_edges)


        return bin_edges


    def raster2png(self, raster_path, out_path, bound=None, bound_crs='EPSG:3857', bins=None,
                   out_size=None, value_factor=1.0, resize_type='scipy', resize_order=1, min_height_value=-1e8,
                   save_geotiff=False):

        # 1. Read the raster (GeoTIFF) into a NumPy array
        with rasterio.open(raster_path) as src:
            if bound is not None:
                # bound is (x_min, x_max, y_min, y_max)
                x_min, y_min, x_max, y_max = bound

                if not isinstance(bound_crs, rasterio.crs.CRS):
                    bound_crs = rasterio.crs.CRS.from_user_input(bound_crs)

                if bound_crs != src.crs.to_epsg():
                    transformer = pyproj.Transformer.from_crs(bound_crs, src.crs, always_xy=True)
                    # Reproject the lower-left and upper-right corners
                    left, bottom = transformer.transform(x_min, y_min)
                    right, top   = transformer.transform(x_max, y_max)
                else:
                    # They are the same CRS
                    left, right, bottom, top = x_min, x_max, y_min, y_max

                # print(left, bottom, right, top)
                # pdb.set_trace()
                # Convert to the format rasterio expects: (left, bottom, right, top)
                # window = rasterio.windows.from_bounds(left=x_min, bottom=y_min, right=x_max, top=y_max, transform=src.transform)
                window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
                height_data = src.read(1, window=window)
            else:
                # Read the full raster
                height_data = src.read(1)

        height_data = height_data * value_factor

        if height_data.shape[0] == 0 or height_data.shape[1] == 0:
            return None, None

        height_data[height_data < min_height_value] = min_height_value

        # 2. Define bins for height categorization
        if bins is None:
            # Compute logarithmic bins between min and max positive heights
            positive_vals = height_data[height_data > 0]

            if positive_vals.size > 0:
                # min_h = max(positive_vals.min(), 1)
                # max_h = positive_vals.max() + 1e-5

                # Create num_bins intervals on a log scale from min_h to max_h
                # old_bin_edges = np.logspace(np.log10(min_h), np.log10(max_h), self.num_bins + 1)
                bin_edges = self.get_histogram_equalized_bins(positive_vals, self.num_bins)

            else:
                # If no positive values (all heights are 0 or missing), just create dummy bins
                bin_edges = np.array([0, 1] * (self.num_bins//2 + 1))
            bins_used = bin_edges  # store for return/output
        else:
            # Use provided bins (ensure numpy array for processing)
            bin_edges = np.array(bins)
            bins_used = bin_edges

        if out_size is not None:
            if resize_type == 'scipy':
                target_h, target_w = out_size
                zoom_h = target_h / height_data.shape[0]
                zoom_w = target_w / height_data.shape[1]
                height_data = scipy.ndimage.zoom(height_data, (zoom_h, zoom_w), order=resize_order)

            elif resize_type == 'resize_sparse':
                height_data = self.resize_sparse_array(height_data, out_size, self.resample_function)

            elif resize_type == 'scipy_rescale':
                zoom = max(*out_size) / max(*height_data.shape)
                height_data = scipy.ndimage.zoom(height_data, (zoom, zoom), order=resize_order)

            else:
                raise ValueError(f'resize type {resize_type} is not supported!')

        # 3. Prepare an RGB image array
        h, w = height_data.shape
        rgba_image = np.zeros((h, w, 4), dtype=np.uint8)

        # 2) Set alpha to fully opaque (255) by default
        rgba_image[..., 3] = 255

        # 3) Mark background pixels (height_data < 0) as transparent
        mask_bg = (height_data <= 0)
        rgba_image[mask_bg, :3] = (0, 0, 0)  # color if you prefer (though alpha=0 means invisible)
        rgba_image[mask_bg, 3] = 128          # make transparent

        # Valid heights: value >= 0 -> apply colormap based on bins
        mask_heights = height_data > 0

        if mask_heights.any():
            # Get colormap and discretize it into N colors
            cmap = cm.get_cmap(self.colormap, self.num_bins)  # ListedColormap with N discrete colors

            if self.colormap_colors is not None:
                colormap_colors = np.array(self.colormap_colors)
            else:
                colormap_colors = (cmap(np.linspace(0, 1, self.num_bins))[:, :3] * 255).astype(np.uint8)

            # Classify each height value into a bin index (0 to num_bins-1)
            values = height_data[mask_heights]
            # Use np.digitize to find bin indices for each value
            # (We exclude the first edge for proper binning since np.digitize returns 1 for values < bin_edges[0])
            bin_indices = np.digitize(values, bin_edges[1:])  # compare against upper edges of bins
            # np.digitize returns 0-based index count; adjust to 0..num_bins-1 range
            bin_indices[bin_indices < 0] = 0
            bin_indices[bin_indices >= self.num_bins] = self.num_bins - 1
            # Assign colors to each pixel based on its bin index
            rgb_values = colormap_colors[bin_indices]
            rgba_image[mask_heights, :3] = rgb_values
        
        # 5. Save the RGB array as a PNG image
        img = Image.fromarray(rgba_image, mode='RGBA')
        img.save(out_path, format='PNG')

        if save_geotiff:
            self.save_image_to_geotiff(rgba_image, bound, out_path.replace('png', 'tif'))

        return img, bins_used

    @staticmethod
    def get_square_roi(min_x, max_x, y):
        """
        Generate the bounding box of a square ROI: (x_min, y_min, x_max, y_max).

        The square has side length (max_x - min_x), and its center is located at:
            ( (min_x + max_x) / 2, y ).

        Parameters
        ----------
        min_x : float
            The minimum X coordinate (left side of the square).
        max_x : float
            The maximum X coordinate (right side of the square).
        y : float
            The Y coordinate of the square's center.

        Returns
        -------
        bbox : tuple[float, float, float, float]
            The bounding box (x_min, y_min, x_max, y_max).
        """
        # 1. Calculate the side length and half the side.
        side_length = max_x - min_x
        half_side = side_length / 2.0

        # 2. The center in X is (min_x + max_x) / 2; center in Y is y.
        center_x = (min_x + max_x) / 2.0
        center_y = y

        # 3. The bounding box in the X dimension is simply [min_x, max_x].
        #    In the Y dimension, it's centered at 'y' with total height == side_length.
        x_min_sq = center_x - half_side  # This equals min_x
        x_max_sq = center_x + half_side  # This equals max_x
        y_min_sq = center_y - half_side
        y_max_sq = center_y + half_side

        return (x_min_sq, y_min_sq, x_max_sq, y_max_sq)

    @staticmethod
    def save_colormap_legend(colormap_colors, bin_edges, out_path, title="Legend"):
        num_bins = len(colormap_colors)
        
        # Set figure size (1:5 height-width ratio)
        fig_width = 20  # Adjust this for wider legends
        fig_height = fig_width / 5  # Ensure 1:5 aspect ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        ax.set_xlim(0, num_bins)
        ax.set_ylim(0, 1)

        # Draw color patches (horizontally)
        for i in range(num_bins):
            color = colormap_colors[i] / 255.0  # Normalize to [0,1]
            rect = mpatches.Rectangle((i, 0), 3, 1, color=color)
            ax.add_patch(rect)

            # Format label (bin range)
            if i < len(bin_edges) - 1:
                label = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
            else:
                label = f">= {bin_edges[i]:.2f}"

            # Place the text below each color patch
            ax.text(i + 0.5, -0.2, label, ha='center', va='top', fontsize=10, rotation=0)

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Add title
        ax.set_title(title, fontsize=12, pad=10)

        # Save the figure
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close to free memory


    def resize_sparse_array(self, input_array, out_size, resample_function):
        """
        Resizes a sparse array by splitting it into grids and applying a resampling function to each grid.

        Parameters:
        - input_array (numpy.ndarray): Input 2D array to be resized.
        - out_size (tuple): (H, W), the shape of the output resized array.
        - resample_function (function): Function to apply to each grid.
                                        This function should take a 2D numpy array and return a single value.

        Returns:
        - resized_array (numpy.ndarray): Resampled array of shape (H, W).
        """
        H_out, W_out = out_size
        H_in, W_in = input_array.shape

        if H_out < H_in and W_out < W_in:

            # Compute grid size
            grid_H = H_in / H_out
            grid_W = W_in / W_out

            # Initialize output array
            resized_array = np.zeros((H_out, W_out), dtype=input_array.dtype)

            for i in range(H_out):
                for j in range(W_out):
                    # Define grid boundaries
                    start_H = int(i * grid_H)
                    end_H = int((i + 1) * grid_H)
                    start_W = int(j * grid_W)
                    end_W = int((j + 1) * grid_W)

                    # Extract the grid
                    grid = input_array[start_H:end_H, start_W:end_W]

                    # Apply the resampling function
                    resized_array[i, j] = resample_function(grid)
        else:
            target_h, target_w = out_size
            zoom_h = H_out / H_in
            zoom_w = W_out / W_in
            resized_array = scipy.ndimage.zoom(input_array, (zoom_h, zoom_w), order=0)

        return resized_array

    def resample_function(self, grid):
        resample_cfg = self.vis_conf.get('resample_cfg', {})
        if resample_cfg.get('sample_type', 'max') == 'max':
            return grid.max()
        elif resample_cfg.get('sample_type', 'max') == 'positive_sum':
            if (grid > 0).sum() == 0:
                return -1
            return grid[grid > 0].sum()
        elif resample_cfg.get('sample_type', 'max') == 'positive_avg':
            if (grid > 0).sum() == 0:
                return -1
            return grid[grid > 0].mean()

    @staticmethod
    def save_image_to_geotiff(image_array, bounds, out_path, crs="EPSG:3857", dtype=None):
        """
        Saves an image array to a GeoTIFF file with the given bounds and CRS.

        Parameters:
        - image_array (numpy.ndarray): Image data of shape (H, W, C) or (H, W).
        - bounds (tuple): Bounding box in format (min_x, min_y, max_x, max_y) in EPSG:3857.
        - out_path (str): Output file path for the GeoTIFF.
        - crs (str): Coordinate reference system (default: "EPSG:3857").
        - dtype (str or None): Data type for raster (default: inferred from array).

        Returns:
        - None (Saves GeoTIFF to `out_path`).
        """
        # Ensure the image array is in (Bands, H, W) format
        if image_array.ndim == 3:  # (H, W, C)
            image_array = np.moveaxis(image_array, -1, 0)  # Convert to (C, H, W)
        elif image_array.ndim == 2:  # (H, W)
            image_array = image_array[np.newaxis, ...]  # Convert to (1, H, W)

        H, W = image_array.shape[1:]  # Extract height and width
        min_x, min_y, max_x, max_y = bounds

        # Compute the affine transformation
        transform = from_bounds(min_x, min_y, max_x, max_y, W, H)

        # Infer dtype if not provided
        if dtype is None:
            dtype = image_array.dtype

        # Define metadata for the GeoTIFF
        metadata = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": image_array.shape[0],  # Number of bands
            "dtype": dtype,
            "crs": rasterio.crs.CRS.from_string(crs),
            "transform": transform
        }

        # Save to GeoTIFF
        with rasterio.open(out_path, "w", **metadata) as dst:
            dst.write(image_array)

        print(f"GeoTIFF saved to {out_path}")
