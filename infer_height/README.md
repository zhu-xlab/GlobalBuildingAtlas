# Height Inference on Planet image on GPU

## Overview

This script performs sliding-window inference on Planet satellite imagery using a pre-trained model.

## Usage
### Directory Setup

Ensure the following:

- The trained model's configuration file (`config.yaml`) and checkpoint file (`checkpoint_best_rmse.pth.tar`) are located in the directory specified by `modelFolder`.
- You have a root directory for storing predictions (`predRootDir`).

### 1. Inference on Individual Mosaics

Use `main.py` for processing specific mosaic files.

- Update `mosaicFilenames` in `main.py` with paths to your Planet mosaic images.
- Run the script with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2... python main.py
```

### 2. Inference on Batch Mosaics

Use `main_mp.py` for batch processing across directories.
- Set the root directory containing Planet mosaics in `procRootDir` within `main_mp.py`.
- Run the script with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2... python main_mp.py
```