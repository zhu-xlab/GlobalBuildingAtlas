# Quality-Based Multi-Source Building Footprint Fusion

## Overview

This project performs quality-based fusion of building footprints from multiple data sources. It processes data into spatial tiles and intelligently merges them based on quality metrics, prioritizing one main source and supplementing it with high-quality data from other sources.

## Usage

### 1. Preprocessing (Optional)
Prepare building footprint datasets (e.g., Microsoft, OSM, Google, 3D Global FP) and organize them into 5°x5° tiles following the Planet mosaic format. Store them in directories as follows:

- `MS_CACHE_DIR`
- `OSM_CACHE_DIR`
- `GOOGLE_CACHE_DIR`
- `GLOBFP_CACHE_DIR`

These directory paths are defined in `utils.py`.

### 2. Administrative Boundaries
Download the [GADM dataset](https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip) and split it into 5°x5° tiles, matching the Planet mosaic grid. Store the tiles in the `district_folder` defined in `main.py` or `main_chunked.py`.

For this, you can refer to `get_boundary_tiles.py`.

### 3. Setup Cache and Output Directories
Create the following directories:
- `cache_dir`: For storing intermediate cropped and processed files
- `output_dirs`: For storing final output files

Both are defined in `main.py` or `main_chunked.py`.

### 4. Input List
Create a text file listing all the input GeoJSON files to be processed, one per line.

### 5. Run the Script

#### For Tiles with Moderate Building Counts (typically <10M buildings):

```bash
python main.py --input_file /path/to/input_list.txt --main_source osm --processes 0
```
- `--main_source`: The primary reference dataset. Typically osm is used outside Africa and South America, where google is preferred.
- `--processes`: Set to 0 to use all available CPU cores.

#### For Tiles with Extremely Large Building Counts (typically >10M buildings):
```bash
python main_chunked.py --input_file /path/to/input_list.txt --main_source osm --processes 0
```

## Principle
The fusion strategy is based on the methodology described in the GlobalBuildingAtlas paper:

Cropping: All building footprint datasets are spatially cropped to match the extents of individual administrative regions. In main_chunked.py, this step is performed in smaller chunks for efficiency.

Quality Evaluation: Each secondary dataset is evaluated against the main source using a combined metric that includes recall and additional non-overlapping building area. The dataset with the best quality is selected as a secondary source.

Selective Merging: Buildings from the main source are preserved in full. Secondary buildings are added only if they do not overlap with any buildings from the main source.

Merging Output: Results from all administrative regions are merged into a single 5°x5° tile file.

## References
GlobalBuildingAtlas Paper (Coming Soon)