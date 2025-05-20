# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch. Dataset4EO works on Linux (Windows and macOS are not officially supported). 

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name earthnets -y
conda activate earthnets
```

**Step 2.** Install required libraries. Core libraries: `torch`, `torchvision`, `torchdata`

```shell
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install torchdata
pip install mmcv-full==1.6.0
pip install prettytable
pip install pycocotools
pip install wandb
```

# Installation


## Install Dataset4EO

```shell
git clone git@github.com:DeepAI4EO/Dataset4EO.git
python -m pip install -e .
```

## Usage

Upon you finish the installation, you can use Dataset4EO like below:

```python
from Dataset4EO.datasets import list_datasets, load, landslide4sense
from torch.utils.data import DataLoader2
from tqdm import tqdm

#list all the supported datasets
print(list_datasets())

#create new dataset object by calling:
datasets_dir = './'
dp = landslide4sense.Landslide4Sense(datasets_dir, split='train')

#Then the corresponding dataset will be downloaded and decompressed automatically

#create a dataloader by calling:
data_loader = DataLoader2(dp.shuffle(), batch_size=4, num_workers=4, shuffle=True, drop_last=True)

#Now, iterating the dataloader for training
for it in tqdm(data_loader):
    print(it) # this will print the filenames of each sample
```