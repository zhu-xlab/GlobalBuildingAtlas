# GlobalBuildingAtlas

## Introduction
In this project, we provide the level of detail 1 (LoD1) data of buildings across the globe.

A overview of the dataset is illustrated bellow:

<img src="figures/overview.png" width="800">

## **License Notice:**  
> ⚠️This dataset is provided in three parts: **ODbL-licensed polygons** (`GBA.ODbLPolygon`), **CC BY-NC 4.0 polygons and LoD1 building models** (`GBA.Polygon` and `GBA.LoD1`), and **CC BY-NC 4.0 height maps** (`GBA.Height`).  
> ⚠️Users may combine these datasets for analysis or downstream applications, but doing so **may create license implications**.  
> ⚠️It is the responsibility of each user to ensure that their use complies with the respective licenses.  
> ⚠️This repository does **not provide legal advice**; users should review the original licenses and, if necessary, consult legal counsel.

## FAQ
### Is Canada / Türkiye / ... in the dataset?
The dataset aims at global coverage, so all countries, territories, and cities should be included. However, due to data quality limitations, some areas may be absent or may not have height attributes. You can check availability using our web viewer.

### Why are the height values incorrect? Why are some buildings missing?
This is a machine-learning–derived product. Errors may occur. Please refer to the [publication](#how-to-cite) for validation results and further details.

### I cannot access the web viewer.
High traffic can occasionally affect the web viewer. We restart the server as needed to maintain access. Please check the bulletin board for maintenance updates.

### Why are there no LoD1 GeoJSON files anymore? Why is the dataset split?
We used some building footprints from ODbL-licensed sources ([OSM](https://www.openstreetmap.org/) and [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints)). According to ODbL, derivatives must also be ODbL-licensed, which conflicts with our PLANET-derived BY-NC data. Therefore, the dataset is split:

- **Part I** – [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.ODbLPolygon): `GBA.ODbLPolygon` contains **only building polygons derived from ODbL-licensed sources**.
- **Part II** – [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1): `GBA.LoD1` contains additional building footprints from other sources, and LoD1 JSON files linking all building polygon features from `GBA.ODbLPolygon` and `GBA.Polygon`.
- **GBA.Height** – [mediaTUM](https://mediatum.ub.tum.de/1782307).

### I still want the LoD1 GeoJSON files. What should I do?
You can follow the README instructions under this [repository](#how-to-use-the-data), or [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1) to derive the final LoD1 GeoJSON files.

⚠️ By deriving the final LoD1 GeoJSON files, you acknowledge the [License Notice](#license-notice).

### What is the CRS of the dataset?
All building polygons are recorded in **EPSG:3857**. Some files in `GBA.ODbLPolygon` on [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.ODbLPolygon) may appear in EPSG:4326 — please treat them as EPSG:3857.

## Access to the Data
### Web Viewer
A web interface for viewing the data is available at: [website](https://tubvsig-so2sat-vm1.srv.mwn.de).

Note: This webviewer is intended for interactive visualization only. Please do not use the WFS service for feature streaming, automated querying, or bulk data extraction. For data access, please use the following dataset release.

### Full Data Download
⚠️ By downloading the data, you agree to the [Terms of Use](https://tubvsig-so2sat-vm1.srv.mwn.de/terms_of_use.html) and acknowledge the [License Notice](#license-notice).

The full data can be downloaded as follows:
- **Part I** - [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.ODbLPolygon), GBA.ODbLPolygon contains ONLY building polygons derived from ODbL-licensed data sources.
- **Part II** - [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1), GBA.LoD1 contains additional building footprints from other data sources, and LoD1 JSON files linking all building polygons features from GBA.ODbLPolygon and GBA.Polygon.
- **GBA.Height** - [mediaTUM](https://mediatum.ub.tum.de/1782307).

## How to Use the Data
1. **Access the representative dataset**  
   - Either in the `representative/` folder from [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1/tree/main/representative) folder of this repository, or via [mediaTUM](https://mediatum.ub.tum.de/1782307).
2. **Identify tiles overlapping your region of interest (RoI)**  
   - For polygons and LoD1 models: use `lod1.geojson` to find the intersecting tiles in `GBA.Polygon` or `GBA.LoD1`.  
   - For heights: use `height_zip.geojson` and `height_tif.geojson` to find intersecting tiles in `GBA.Height`.
3. **Download required tiles**  
   - **ODbL polygons**: download from [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.ODbLPolygon) and save under:

     ```
     ./ODbLPolygon
     ```
   - **LoD1 polygons and other data**: download from [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1), and save under:
     ```
     ./Polygon
     ./LoD1
     ```
4. **Run the enrichment script provided at [HuggingFace](https://huggingface.co/datasets/zhu-xlab/GBA.LoD1/blob/main/produce_lod1.py)**  
   ```bash
   python produce_lod1.py
    ```
   - You may also specify folder paths at your choice.
    ```bash
    python produce_lod1.py \
        --odbl_root /path/to/odbl \
        --polygon_root /path/to/polygon \
        --json_root /path/to/json \
        --output_root /path/to/output
    ```
   - The script will read the two GeoJSON folders and the JSON folder, merge the properties, and add height and var fields.

5. **Output**
   - LoD1 GeoJSON files will be written under
    ```
    ./LoD1_GeoJSON
    ```
    or the folder you specified.
    - If you are only interested in GBA.Polygon, you can ignore the `height` and `var` fields.

## Development Code
### Global Building Polygon Generation using Satellite Data (Sec. 4.3)
For codes related to building map extraction, regularization, polygonization, and simplification, i.e., generating building polygons from satellite images (Sec. 4.3.2, Sec. 4.3.3, and Sec. 4.3.4), please refer to `./im2bf`.

### Global Building Height Estimation (Sec. 4.4)
1. For codes related to monocular height estimation using HTC-DC Net (Sec. 4.4.2), please refer to `./im2bh`.
2. For codes related to the global inference and uncertainty quantification (Sec. 4.4.3), please refer to `./infer_height`

### Global LoD1 Building Model Generation (Sec. 4.5)
1. For codes related to quality-guided building polygon fusion (Sec. 4.5.1), please refer to `./fuse_bf`.
2. For codes related to LoD1 building model generation (Sec. 4.5.2), please refer to `./make_lod1`.

## Visualization Code
For codes to reproduce the plots in the manuscript, please refer to `./make_plots`.

## Code License
MIT with Commons Clause (no commercial use allowed). See [LICENSE](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/main/LICENSE).

## How to cite
If you find this dataset helpful in your work, please cite the following paper.
```
@Article{essd-17-6647-2025,
AUTHOR = {Zhu, X. X. and Chen, S. and Zhang, F. and Shi, Y. and Wang, Y.},
TITLE = {GlobalBuildingAtlas: an open global and complete dataset of building polygons, heights and LoD1 3D models},
JOURNAL = {Earth System Science Data},
VOLUME = {17},
YEAR = {2025},
NUMBER = {12},
PAGES = {6647--6668},
URL = {https://essd.copernicus.org/articles/17/6647/2025/},
DOI = {10.5194/essd-17-6647-2025}
}
```
