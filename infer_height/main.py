#!/usr/bin/env python

"""planet_inferencing_sr_gpu.py: inferencing on GPU with Planet data.

__author__      :  "Yilei Shi"
__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__maintainer__  :  "Yilei Shi"
__email__       :  "yilei.shi@tum.de"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import torch
import numpy as np
import os
import cv2
import timeit
import yaml
import warnings
from glob import glob

from utils import *

def pred(gpuRank, modelFolder, mosaicFilename, bfsTIF, useBasemap=False, rgb=True):
    modelConfigFile = os.path.join(modelFolder, "config.yaml")
    cfgs = load_yaml(modelConfigFile)
    model = get_model(cfgs)
    chkptFile = glob(os.path.join(cfgs["experiment_dir"], "checkpoint_best_rmse.pth.tar"))[0]
    chkpt = torch.load(chkptFile)
    model.load_state_dict(chkpt["state_dict"])

    varsTIF = bfsTIF.replace("_sr_ss.tif", "_sr_var.tif")
    planet_infer_sliding_window(gpuRank, model, mosaicFilename, bfsTIF, varsTIF, useBasemap, rgb)


if __name__ == "__main__":
    gpuRank = "0"
    modelFolder = "/data/chen/repos/GBH/checkpoints/adabins_htc/230920_094641/"

    mosaicFilenames = glob(os.path.join("parallelTest", "*", "*", "*", "*.tif"))

    for mosaicFilename in mosaicFilenames:
        continent = mosaicFilename.split("/")[1]
        tile = mosaicFilename.split("/")[3]
        predRootDir = os.path.join("output/", continent.upper(), tile)
        os.makedirs(predRootDir, exist_ok=True)

        bfsTif = os.path.join(predRootDir, os.path.basename(mosaicFilename).replace("_mosaic_tile.tif", "_ss.tif"))
        pred(gpuRank, modelFolder, mosaicFilename, bfsTif, rgb=False)
    
    
