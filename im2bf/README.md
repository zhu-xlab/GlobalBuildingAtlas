# Global Building Polygon Generation
We provide the codes for developing our global building polygon generation results.
Note that due to license limitation, we are not able to provide PlanetScope data and pretrained weights based on the data.
In this repo, we provide the codes for training the regularization networks and the following polygonization process described in Sec. 4.3.3 and 4.3.4 in the original paper.

## Installation
Follow the bellow steps to reproduce the 
1. A .yaml file is provided for creating the conda environment. Run `conda env create -f environment.yaml` to create the environment.
2. Install the two repo by:
   ```
   pip install -e Dataset4EO
   pip install -e GBA_Poly
   ```

## Train the Regularization Networks
Go into the GBA_Poly folder and run:
```
python tools/train.py configs/gba_poly/regu_upernet_convnext-t_80k.py
```
to train the regularization networks introduced in Sec. 4.3.3.
We provide sample building vector data in `GBA_Poly/data/shapes` folder to train the network. If you would like to extend the dataset, simply put more .shp files into the folder.

We also provide pretrained weights with the full building vector data we have in `work_dirs/regu_upernet_convnext-t_80/iter_80000.pth`.

## Conduct Polygonization with Binary Building Masks
To conduct the building regularization, polygonization and simplification altoghether with a binary building mask, one can run:
```
python tools/test.py configs/gba_poly/inference_polygonization.py --format-only
```
A sample data at `GBA_Poly/data/masks/sample.tif` will be loaded and processed. Output vector data will be generated in `GBA_Poly/data/outputs`.

