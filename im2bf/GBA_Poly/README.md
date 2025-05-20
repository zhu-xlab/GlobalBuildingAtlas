<div align="center">
  <img src="resources/rsifewshot-logo.png" width="500"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rsifewshot)](https://pypi.org/project/rsifewshot/)
[![PyPI](https://img.shields.io/pypi/v/rsifewshot)](https://pypi.org/project/rsifewshot)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://rsifewshot.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/rsifewshot/workflows/build/badge.svg)](https://github.com/open-mmlab/rsifewshot/actions)
[![codecov](https://codecov.io/gh/open-mmlab/rsifewshot/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/rsifewshot)
[![license](https://img.shields.io/github/license/open-mmlab/rsifewshot.svg)](https://github.com/open-mmlab/rsifewshot/blob/master/LICENSE)

[📘Documentation](https://rsifewshot.readthedocs.io/) |
[🛠️Installation](https://rsifewshot.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://rsifewshot.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://rsifewshot.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/rsifewshot/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

rsifewshot is an open source few shot learning toolbox based on PyTorch. It is a part of the [OpenMMLab](https://open-mmlab.github.io/) project.

The master branch works with **PyTorch 1.5+**.
The compatibility to earlier versions of PyTorch is not fully tested.

<div align="left">
  <img src="resources/demo.png"/>
</div>

### Major features

- **Support multiple tasks in Few Shot Learning**

  MMFewShot provides unified implementation and evaluation of few shot classification and detection.

- **Modular Design**

  We decompose the few shot learning framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in few shot classification and detection.

## What's New

v0.1.0 was released in 24/11/2021.
Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation & Dataset Preparation

MMFewShot depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Please refer to [install.md](/docs/en/install.md) for installation of MMFewShot and [data preparation](tools/data/README.md) for dataset preparation.

## Getting Started

If you are new of few shot learning, you can start with [learn the basics](docs/en/intro.md).
If you are familiar with it, check out [getting_started.md](docs/en/get_started.md) for the basic usage of rsifewshot.

Refer to the below tutorials to dive deeper:

- Few Shot Classification

  - [Overview](docs/en/classification/overview.md)
  - [Config](docs/en/classification/customize_config.md)
  - [Customize Dataset](docs/en/classification/customize_dataset.md)
  - [Customize Model](docs/en/classification/customize_models.md)
  - [Customize Runtime](docs/en/classification/customize_runtime.md)

- Few Shot Detection

  - [Overview](docs/en/detection/overview.md)
  - [Config](docs/en/detection/customize_config.md)
  - [Customize Dataset](docs/en/detection/customize_dataset.md)
  - [Customize Model](docs/en/detection/customize_models.md)
  - [Customize Runtime](docs/en/detection/customize_runtime.md)

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).
Supported algorithms:

<details open>
<summary>Classification</summary>

- [x] [Baseline](configs/classification/baseline/README.md) (ICLR'2019)
- [x] [Baseline++](configs/classification/baseline_plus/README.md) (ICLR'2019)
- [x] [NegMargin](configs/classification/neg_margin/README.md) (ECCV'2020)
- [x] [MatchingNet](configs/classification/matching_net/README.md) (NeurIPS'2016)
- [x] [ProtoNet](configs/classification/proto_net/README.md) (NeurIPS'2017)
- [x] [RelationNet](configs/classification/relation_net/README.md) (CVPR'2018)
- [x] [MetaBaseline](configs/classification/meta_baseline/README.md) (ICCV'2021)
- [x] [MAML](configs/classification/maml/README.md) (ICML'2017)

</details>

<details open>
<summary>Detection</summary>

- [x] [TFA](configs/detection/tfa/README.md) (ICML'2020)
- [x] [FSCE](configs/detection/fsce/README.md) (CVPR'2021)
- [x] [AttentionRPN](configs/detection/attention_rpn/README.md) (CVPR'2020)
- [x] [MetaRCNN](configs/detection/meta_rcnn/README.md) (ICCV'2019)
- [x] [FSDetView](configs/detection/fsdetview/README.md) (ECCV'2020)
- [x] [MPSR](configs/detection/mpsr/README.md) (ECCV'2020)

</details>

## Contributing

We appreciate all contributions to improve rsifewshot. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/rsifewshot/blob/main/.github/CONTRIBUTING.md) in MMFewShot for the contributing guideline.

## Acknowledgement

rsifewshot is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{rsifewshot2021,
    title={OpenMMLab Few Shot Learning Toolbox and Benchmark},
    author={rsifewshot Contributors},
    howpublished = {\url{https://github.com/open-mmlab/rsifewshot}},
    year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
- [MMFewShot](https://github.com/open-mmlab/rsifewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration):  OpenMMLab Generative Model toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMlab deep learning model deployment toolset.
