#!/usr/bin/env python

"""planet_inferencing_parallelProc.py: parallel processing of each step with Planet data.

__author__      :  "Yilei Shi"
__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__maintainer__  :  "Yilei Shi"
__email__       :  "yilei.shi@tum.de"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import os
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
from utils import *

def planet_infer_taskDecompositionPred(csvFilename, rid, numProc=8):

    nTileProcessed = 0
    nTileMosaiced = 0
    nTilePred = 0

    gufDir = []

    tileDir = []
        
    df = pd.read_csv(csvFilename)
    listLines = [list(row) for row in df.values]

    for ind, listLine in enumerate(listLines):

        data = listLine

        if (data[3] != 0):
            nTileProcessed += 1
            if (data[5] == 0):
                nTileMosaiced += 1
                gufDir.append(data[1])
                tileDir.append(data[2])
            if (data[5] == 1):
                nTilePred += 1

    minTasks = nTileMosaiced//numProc
    restTasks = nTileMosaiced%numProc
            
    numTasks = np.ones(numProc) * minTasks

    for ind in range(restTasks):
        numTasks[ind] += 1
                
    indProc = []
    indProc.append(0)
                
    for ind in range(numProc):
        indProc.append(int(indProc[ind*2] + numTasks[ind]))
        indProc.append(int(indProc[ind*2] + numTasks[ind]))

    del indProc[-1]
        
    return gufDir[indProc[2*(rid-1)]:indProc[2*(rid-1)+1]], tileDir[indProc[2*(rid-1)]:indProc[2*(rid-1)+1]]


def planet_infer_predParallel(csvFilename, modelFilename, modelCfgs, procRootDir, predRootDir, np):
    df = pd.read_csv(csvFilename)

    pid = mp.current_process()._identity[0]
    rid = pid%np
    gpuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rid]

    gufDirs, tileDirs = planet_infer_taskDecompositionPred(csvFilename, rid, np)

    for ind, tileDir in enumerate(tileDirs):
        satFilename = procRootDir + 'mosaic/' + gufDirs[ind] + '/' + tileDir + '_sr_mosaic_tile.tif'
        runStat = planet_infer_sliding_window_mp(gpuid, modelFilename, modelCfgs, procRootDir, predRootDir, gufDirs[ind], tileDir)
        if runStat == 0:
            pd_ind = df.index[df["tileDirName"] == tileDir].tolist()[0]
            df.at[pd_ind, "predStat"] = 1

            df.to_csv(csvFilename, index=False)
            
    return 0

def initalize_csvFile(csvFilename, procRootDir, predRootDir):
    sr_mosaic_files = glob(os.path.join(procRootDir, "*", "*_sr_mosaic_tile.tif"))
    
    gufDirNames = []
    tileDirNames = []
    procStats = []
    errorCodes = []
    predStats = []

    for sr_mosaic_file in sr_mosaic_files:
        paths = sr_mosaic_file.split('/')
        gufDir = paths[-2]
        gufDirNames.append(gufDir)
        tileDir = paths[-1].split("_sr_mosaic_tile")[0]
        tileDirNames.append(tileDir)
        procStats.append(1)
        errorCodes.append(0)
        output_file = predRootDir + gufDir + '/' + tileDir + '_sr_ss.tif'
        if os.path.exists(output_file):
            predStats.append(1)
        else:
            predStats.append(0)

    df = pd.DataFrame(
        dict(
            gufDirName=gufDirNames,
            tileDirName=tileDirNames,
            procStat=procStats,
            errorCode=errorCodes,
            predStat=predStats
        )
    )

    df.to_csv(csvFilename)

for continent in ["nordamerica", "europe", "oceania", "southamerica", "africa", "asiaeast", "asiawest"]:
    csvFilename = f"/data/chen/track/{continent}.csv"
    modelFilename = '/data/chen/repos/GBH/checkpoints/adabins_htc/230920_094641/checkpoint_best_rmse.pth.tar'
    modelConfig = '/data/chen/repos/GBH/checkpoints/adabins_htc/230920_094641/config.yaml'
    modelCfgs = load_yaml(modelConfig)
    procRootDir = f"/data/chen/mosaics/{continent}/mosaic/"
    predRootDir = f"/data/chen/outputs/{continent}/"

    numGPU = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


    for i in range(numGPU):
        p = mp.Process(target=planet_infer_predParallel, args=(csvFilename, modelFilename, modelCfgs, procRootDir, predRootDir, numGPU))
        p.start()
    break
