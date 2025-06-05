# GlobalBuildingAtlas

## Introduction
In this project, we provide the level of detail 1 (LoD1) data of buildings across the globe.

A overview of the dataset is illustrated bellow:

<img src="figures/overview.png" width="800">


## Access to the Data
### Web Feature Service (WFS)
A WFS is provided so that one can access the data using other websites or GIS softwares such as QGIS and ArcGIS.

Url: `https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows?` (Temporarily closed due to overwhelming number of visits)

### Web Viewer
A web interface for viewing the data is available at: [website](https://tubvsig-so2sat-vm1.srv.mwn.de) (Temporarily closed due to overwhelming number of visits)

### Full Data Download
The full data can be downloaded from [mediaTUM](https://mediatum.ub.tum.de/178230) (Open soon. Currently under review)

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
