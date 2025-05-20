#!/usr/bin/env python

"""planet_inferencing_utils.py: utilities for inferencing with Planet Data.

__author__      :  "Yilei Shi"
__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__maintainer__  :  "Yilei Shi"
__email__       :  "yilei.shi@tum.de"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import os
import sys
import warnings
import time
import timeit
import gdal
import numpy as np
import itertools
import yaml
import torch
import cv2
        
def planet_infer_readTiff(tiffFilename):

    imgFID = gdal.Open(tiffFilename)
    img = imgFID.ReadAsArray()
    
    return img 

def planet_infer_writeTiff(inTiff, outTiff, outData):
    inDs = gdal.Open(inTiff)
    
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize

    outDs = gdal.GetDriverByName('GTiff').Create(outTiff, cols, rows, 1, gdal.GDT_Float32)
                                                  
    outband = outDs.GetRasterBand(1)
    outband.WriteArray(outData, 0, 0)
    outband.SetNoDataValue(-1)

    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())

    outband.FlushCache()


def planet_infer_swCoords(img, step=64, window_size=(256,256)):

    H = img.shape[1]
    W = img.shape[2]
    
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, H, step):
        if x + window_size[0] > H:
            x = H - window_size[0]
        for y in range(0, W, step):
            if y + window_size[1] > W:
                y = W - window_size[1]
            yield x, y, window_size[0], window_size[1]

            
def planet_infer_swCount(img, step=64, window_size=(256, 256)):

    H = img.shape[1]
    W = img.shape[2]
    
    """ Count the number of windows in an image """
    nSW = 0
    for x in range(0, H, step):
        if x + window_size[0] > H: 
            x = H - window_size[0]
        for y in range(0, W, step):
            if y + window_size[1] > W:
                y = W - window_size[1]
            nSW += 1
            
    return nSW     

def planet_infer_grouper(n, iterable):
    
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
        
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
        
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
            
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
    
def planet_infer_normalize_ahcUpSolo(filename, mean, std, rgb=False):
    img = planet_infer_readTiff(filename)[:3]

    if not rgb:
        img = img[::-1, :, :]
    
    imgStat = 1
      
    imgC, imgH, imgW = img.shape
    imgOut = np.zeros((imgC, imgH, imgW), dtype=np.float32)

    for nC in range(imgC):
        data = img[nC, :, :]
        countzero = np.count_nonzero(data)
        if (countzero == 0):
            imgStat = 0
            return imgOut, imgStat
    imgOut = (img - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]

    return imgOut, imgStat

def planet_infer_grouper(n, iterable):
    
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # suppress the warning caused by yaml.safe_load, because we need objects in .yaml file to be read.
            try:
                return yaml.load(f)
            except:
                return yaml.load(f, Loader=yaml.FullLoader)
            
def get_model(cfgs):
    if cfgs["model"] == "unet":
        from baselines.unet import UNet
        model = UNet(cfgs)
    elif cfgs["model"] == "adabins_htc":
        from baselines.adabins_htc import UBins
        model = UBins(cfgs)
    return model

def planet_infer_sliding_window(gpuRank, model, mosaicFilename, bfsTIF, varsTIF, useBasemap=False, rgb=False, bS=64):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuRank #os.environ["CUDA_VISIBLE_DEVICES"].split(",")[gpuRank]

    if useBasemap:
        image_stats_file = 'data/gbh/basemap_stats.pickle'
    else:
        image_stats_file = 'data/gbh/image_stats.pickle'

    mean, std = torch.load(image_stats_file)
    mean = np.asarray(mean)
    std = np.asarray(std)

    filebasename = os.path.basename(mosaicFilename)
    img, imgStat = planet_infer_normalize_ahcUpSolo(mosaicFilename, mean, std, rgb) # rewrite
    if imgStat:
        model.cuda()
        model.eval()

        inference_startTime = timeit.default_timer()
        print('-----------  ' + filebasename + '  ------------')
    
        pred = np.zeros(img.shape[1:] + (1,), dtype=np.float32)
        count = np.zeros(img.shape[1:] + (1,), dtype=np.float32)
        pred_sqr = np.zeros(img.shape[1:] + (1,), dtype=np.float32)
        numP = planet_infer_swCount(img)
    
        with torch.no_grad():
            for idB, coords in enumerate(planet_infer_grouper(bS, planet_infer_swCoords(img, step=64, window_size=(256, 256)))):
                imgP = [np.copy(img[:, x:x+w, y:y+h]) for x,y,w,h in coords]
                imgP = np.asarray(imgP)
                imgP = torch.from_numpy(imgP).float().cuda()
                outs = model(imgP)
                outs_np = outs.detach().cpu().numpy()

                for out, (x, y, w, h) in zip(outs_np, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out
                    count[x:x+w, y:y+h] += 1
                    pred_sqr[x:x+w, y:y+h] += out*out

                progress_bar(idB, numP//bS, 'numBatch: (%d/%d)' % ((idB+1), numP//bS))

        pred = np.where(count, pred/count, -1).squeeze(-1)
        pred_sqr = np.where(count, pred_sqr/count, -1).squeeze(-1)
        var = pred_sqr - pred * pred
        cv2.imwrite(bfsTIF, pred.astype(np.float32))
        cv2.imwrite(varsTIF, var.astype(np.float32))
        planet_infer_writeTiff(mosaicFilename, bfsTIF, pred)
        planet_infer_writeTiff(mosaicFilename, varsTIF, var)
            
        inference_endTime = timeit.default_timer()
        print("-- inferencing : " + str(inference_endTime - inference_startTime) + " s")


def planet_infer_sliding_window_mp(gpuRank, modelFilename, modelCfgs, procRootDir, predRootDir, gufDir, tileDir, rgb=False, bS=48):
    try:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=gpuRank

        predDir = predRootDir + gufDir + '/'
        os.makedirs(predDir, exist_ok=True)

        mosaicFilename = procRootDir + gufDir + '/' + tileDir + '_sr_mosaic_tile.tif'

        image_stats_file = 'image_stats.pickle'
        mean, std = torch.load(image_stats_file)
        mean = np.asarray(mean)
        std = np.asarray(std)

        bfsTIF = predDir + tileDir + '_sr_ss.tif'
        varsTIF = predDir + tileDir + '_sr_var.tif'
        img, imgStat = planet_infer_normalize_ahcUpSolo(mosaicFilename, mean, std, rgb) # rewrite
        
        if imgStat:
            model = get_model(modelCfgs)
            checkpoint = torch.load(modelFilename)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()

            inference_startTime = timeit.default_timer()
            print('-----------  ' + tileDir + '  ------------')
        
            pred = np.zeros(img.shape[1:] + (1,), dtype=np.float32)
            count = np.zeros(img.shape[1:] + (1,), dtype=np.float32)
            pred_sqr = np.zeros(img.shape[1:] + (1,), dtype=np.float32)
            numP = planet_infer_swCount(img)
        
            with torch.no_grad():
                for idB, coords in enumerate(planet_infer_grouper(bS, planet_infer_swCoords(img, step=128, window_size=(256, 256)))):
                    imgP = [np.copy(img[:, x:x+w, y:y+h]) for x,y,w,h in coords]
                    imgP = np.asarray(imgP)
                    imgP = torch.from_numpy(imgP).float().cuda()
                    outs = model(imgP)
                    outs_np = outs.detach().cpu().numpy()

                    for out, (x, y, w, h) in zip(outs_np, coords):
                        out = out.transpose((1,2,0))
                        pred[x:x+w, y:y+h] += out
                        count[x:x+w, y:y+h] += 1
                        pred_sqr[x:x+w, y:y+h] += out*out

                    progress_bar(idB, numP//bS, 'numBatch: (%d/%d)' % ((idB+1), numP//bS))


            pred = np.where(count, pred/count, -1).squeeze(-1)
            pred_sqr = np.where(count, pred_sqr/count, -1).squeeze(-1)
            var = pred_sqr - pred * pred
            cv2.imwrite(bfsTIF, pred.astype(np.float32))
            cv2.imwrite(varsTIF, var.astype(np.float32))
            planet_infer_writeTiff(mosaicFilename, bfsTIF, pred)
            planet_infer_writeTiff(mosaicFilename, varsTIF, var)
                
            inference_endTime = timeit.default_timer()
            print("-- inferencing : " + str(inference_endTime - inference_startTime) + " s")
        return 0
    except:
        return 1
