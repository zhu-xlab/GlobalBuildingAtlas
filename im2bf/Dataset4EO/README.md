<div  align="center">    
 <img src="resources/datasets4eo.png" width = "400" height = "130" alt="segmentation" align=center />
</div>


![example workflow](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)

[Homepage of the project](https://earthnets.nicepage.io/)

## Composable data loading based on [TorchData](https://github.com/pytorch/data)
It aims to provide composable Iterable-style and Map-style building blocks called DataPipes that work well out of the box with the PyTorch's DataLoader. It contains functionality to reproduce many different datasets in TorchVision and TorchText, namely including loading, parsing, caching, and several other utilities (e.g. hash checking).

Todo List

- [x] Re-organize mroe than 400 datasets in Remote sensing cimmunity in a task-oriented way;
- [x] preprocess the datasets in a AI-ready style;
- [x] implementing dataset classes for downloading;
- [x] supporting for heigh-level repos for specific tasks: obejct detection, segmentation and so forth;
- [x] supporting dataloaders in a easy-to-use way for custom projects;
- [x] benchmarking cutting-edge CV backbones and models on RS data;
- [ ] implementing the random sampler for geo-spatial datasets;

## Supported datasets:

- [x] [DFC2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)
- [x] [LandSlide4Sense](https://www.iarai.ac.at/landslide4sense/)
- [x] [Eurosat](https://github.com/phelber/EuroSAT#)
- [x] [AID](https://captain-whu.github.io/AID/)
- [x] [DIOR](http://www.escience.cn/people/JunweiHan/DIOR.html)
- [x] [DOTA 2.0](https://captain-whu.github.io/DOTA/index.html)
- [x] [fMoW](https://github.com/fMoW/dataset)
- [x] [GeoNRW](https://github.com/gbaier/geonrw)
- [x] [LoveDA](https://github.com/Junjue-Wang/LoveDA)
- [x] [NWPU_VHR10](https://github.com/chaozhong2010/VHR-10_dataset_coco)
- [x] [RSUSS](https://github.com/EarthNets/RSI-MMSegmentation)
- [x] [BigEarthNet](https://bigearth.net/)
- [x] [SEASONET](https://zenodo.org/record/5850307#.Y0cayXbP1D8)
- [x] [SSL4EO_S12](https://github.com/zhu-xlab/SSL4EO-S12)
- [x] [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- [x] [Satlas](https://satlas-pretrain.allen.ai)

Continually updating

## How to use

### Install newest versions of torch and torchdata
```shell
sh install_requirements.sh
```

### Install Dataset4EO
```shell
git clone git@github.com:DeepAI4EO/Dataset4EO.git
python -m pip install -e .
```

### Then it can be used by
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
    print(it)
```

# Add Transformations
```python
from Dataset4EO import transforms

tfs = transforms.Compose(transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.RandomResizedCrop((128, 128), scale=[0.5, 1]))
                                 
ndp = ndp.map(tfs)
```


# Contribution Guidelines
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# Citation
```BibTeX
@article{earthnets4eo,
    title={EarthNets: Empowering AI in Earth Observation},
    author={Zhitong Xiong, Fahong Zhang, Yi Wang, Yilei Shi, Xiao Xiang Zhu},
    journal = {arXiv:2210.04936},
    year={2022}
}
```

# Acknowledgment
We thank the following open dataset collections:

1. https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760
2. https://github.com/chrieke/awesome-satellite-imagery-datasets
3. https://github.com/zhangbin0917/Awesome-Remote-Sensing-Dataset
4. https://github.com/robmarkcole/satellite-image-deep-learning#lists-of-datasets
5. https://eod-grss-ieee.com/dataset-search
6. https://mlhub.earth/datasets
7. https://github.com/biasvariancelabs/aitlas-arena
8. https://github.com/Agri-Hub/Callisto-Dataset-Collection
9. https://github.com/wenhwu/awesome-remote-sensing-change-detection
10. https://github.com/pubgeo/datasets
11. http://eodata.bvlabs.ai/#/
12. https://github.com/MinZHANG-WHU/Change-Detection-Review
13. http://datahub.geocradle.eu/search/type/dataset
14. https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm#remote
15. https://www.cosmiqworks.org/projects/
16. https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-1-W2/1/2019/isprs-archives-XLII-1-W2-1-2019.pdf
