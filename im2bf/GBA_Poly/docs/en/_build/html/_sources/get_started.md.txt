## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [Dataset4EO](https://github.com/EarthNets/Dataset4EO)
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


## Installation

(Install Prerequisites)

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name earthnets -y
conda activate earthnets
```

**Step 2.** Install required libraries. Core libraries: `torch`, `torchvision`, `torchdata`.

```shell
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install torchdata
pip install mmcv-full==1.6.0
pip install prettytable
pip install pycocotools
pip install wandb
```

(Install Dataset4EO)

**Step 3.** Install Dataset4EO.

```shell
git clone git@github.com:DeepAI4EO/Dataset4EO.git
python -m pip install -e .
```

(Install RSI-Segmentation)

**Step 4.** Install RSI-Segmentation.

```shell
git clone https://github.com/EarthNets/RSI-Segmentation.git
cd RSI-Segmentation
pip install -e .
```
