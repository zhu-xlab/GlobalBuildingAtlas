# Figure Creation for Global3D paper
## Figure 1: Workflow of the Proposed Method
Editable links available at: [Fig. 1](https://drive.google.com/file/d/1Lq7RCVfeCegxCEKZbbypuSE_mIQztAcP/view)

## Figure 2: Overview of the Product
Editable links available at: [Fig. 2](https://drive.google.com/file/d/1RFDTwzjvY_x6xm9jXoETj4lF1dTkWUnl/view?usp=sharing)

The figure is created according to the following processes:
### Top Part
To generate the figure for building number, area and volume distribution, simply run: 
```
python plot_bar_chart_statis.py
```
Change the input and output location accordingly based on your access to GLOBAL3D_DATA_DIR

Change `stat_name`, `title` and `color_tone` variables to generate distribution of building number, area, and volume, respectively.

To generate the figure for height RMSE and volume RMSE, simply run:
```
python plot_bar_chart_rmse.py
```
Change the out_path accordingly to save to another location.

### Middle Center Part
To generate the map, simply run:
```
python plot_tif_lod1.py
```
Important variables to change:

1. `tif_path` and `out_dir`: change base on you access to GLOBAL3D_DATA_DIR
2. `default_bins`: volume ranges for generating the colormap. Set it to None to calculate automatically.
3. `num_bins` in line 85: change the number to use more colors.
4. `bound_dict` and `roi_list`: For each roi, a predefined boundary will be used to generate the figure. 'Globe' roi corresponds to Fig. 2.
5. `out_size`: output size of the figure.

### Middle Left and Right Parts
1. LoD1 samples and rois are available at: `/datastore01/DATA/3D/PLANET/Global3D/figures/plots_samples`
2. There are two rois, LoD1 geojsons and height maps in the dir.
3. The 3D buildings are rendered based on Qgis2threejs plugin of QGIS.
4. Colormap is decided based on the volume distribution.

### Bottom Part
The corresponding code is still 
```
python plot_tif_lod1.py
```
Uncomment cities in `roi_list`, and also the corresponding boundary in `bound_dict` to allow batch processing of plenty of cities.

## Figure 3: Radar Chart for Comparison Methods
Editable links available at: [Fig. 3](https://drive.google.com/file/d/11JcStnas268q2xTDlLobfdl9MVq-CXSl/view?usp=sharing)

To generate the chart, run:
```
python make_radar_chart.py
```
Change `plot_legened` to control if the legend will be plotted.

## Figure 4 and 5: Visual Comparison
Editable links available at [Fig. 4 and 5](https://drive.google.com/file/d/1Qcn4_Y3ycFo2ilgk9w1WTqUaAg1P7UEb/view?usp=sharing)

To generate the map, run:
```
python plot_tif_samples.py
```
1. Change `roi_list` to define the target cities.
2. Change `colormap_colors` to use different color maps. Change `num_bins` accordingly if needed.

## Figure 6: EU-wide by Country Population-Volume Regression
To generate the figure, run:
```bash
python plot_europe_regression.py
```
It uses the following two data files.
1. `europe_regression_metrics.json`, which stores the regression metrics of each country in Europe. 
2. `europe_population_building_volume.geojson`, which stores the 1km by 1km grid population and building volume information in Europe. The file is too large and thus, only stored at `/datastore01/DATA/3D/PLANET/Global3D/analysistools/europe_population_building_volume.geojson`.

## Figure 7: EU-wide Building Volume per Capita and Correlation Coefficient Bar Plot
To generate the figure, run:
```bash
python plot_europe_bypc_cc.py
```
It uses `europe_regression_metrics.json`, which stores the building volumes and populations of each country in Europe.

### Figure 8: Worldwide Population-Volume Regression + Worldwise Building Volume per Capita Bar Plot
To generate the figure, run:
```bash
python plot_global_regression.py
```
It uses `global_population_building_volume.json`, which stores the building volumes and populations of each country.

### Figure 9: Worldwide GDP-Volume and GDP-Area Regression
To generate the figure, run:
```bash
python plot_sdg_indicator.py
```
It uses `global_population_building_volume.json`, which stores the building volumes, areas, gdp per capita and populations of each country.

### Figure 10: Distribution of Training and Testing Data
Editable links available at [Fig. 10](https://drive.google.com/file/d/1J7Xm3vC0My0OtJGA6lvfFgM7yhTuTnjV/view?usp=sharing)

Geojsons files for rois of the cities is placed at:
```
/datastore01/DATA/3D/PLANET/Global3D/figures/data_curation.zip
```
### Figure 11: Distribution of Building Volume, and by Country Bar Plot
To generate the figure, run:
```bash
cd analysistools
python plot_global_volume_distribution.py
```
It uses `volume_by_country.json`, which stores the building volumes of each country.

### Figure 12: Contribution of Different Data Sources
Editable links available at [Fig. 12 and 13](https://drive.google.com/file/d/1B0hjR-OWEzH7m7egS6ST0Jkw63x6g8rN/view?usp=sharing)

To produce the figure, run:
```
python plot_bar_chart_contribution.py
```
Change `stat_type` and `title` to plot building count and building area.

### Figure 13: Examples of Fused Building Polygon Results
Editable links available at [Fig. 12 and 13](https://drive.google.com/file/d/1B0hjR-OWEzH7m7egS6ST0Jkw63x6g8rN/view?usp=sharing)

Geojson samples are placed at `/datastore01/DATA/3D/PLANET/Global3D/figures/samples_for_contribution.zip`

