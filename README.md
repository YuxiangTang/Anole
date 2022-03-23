# Anole

An End-to-End Code Library and Public Platform for Color Constancy 

Color constancy is a technology closely related to image color quality, which is dedicated to making cameras record real colors of the whole world. In order to promote the development of color constancy, we are committed to open source the first color constancy codebase and designing a lightweight, easy-to-use, and easy-to-expand platform, which can:

- Reproduce SOTA results quickly, 
- Provide a fair evaluation framework,
- Provide mature cutting-edge model and color constancy toolchains.

The library is organized into four main modules:

- RAW Image Processing [[anole/raw_processing](https://github.com/YuxiangTang/Anole/tree/master/anole/raw_processing)]: Mainly includes all kinds of RAW image pre-and post-processing functions, such as black/white level correction, de-mosaic, color space conversion, gamma correction, tone curve correction, etc. These functions have strong versatility in RAW image processing and are also the important processing nodes in the camera pipeline.
- Color Constancy Modelzoo [[anole/model](https://github.com/YuxiangTang/Anole/tree/master/anole/model)]: Deep-learning research has a core obstacle --"baseline is difficult to reproduce". Therefore, a long-term goal of this codebase is to unify the popular and commonly used baseline under one set of frameworks, and to provide a quick and less code way to build and reproduce the model on various ends. 
- Unified Evaluation Framework [[anole/training](https://github.com/YuxiangTang/Anole/tree/master/anole/training)]: It is difficult for people without camera processing experience to get started with color constancy, and some specifications are not accurately transmitted to various researchers, which leads to irregular use of data sets and inconsistent evaluation standards. To solve this problem, this codebase provides a unified testing framework and system, so as to standardize and unify the indicators of each dataset.
- Experimental Management [[./config](https://github.com/YuxiangTang/Anole/tree/master/config)]: In this code base, you can manage the experiment by generating a report over Hydra and visualization technology after each experiment.

## Getting Started

1. Setup Environment

```bash
pip install -r requirements.txt
```

2. Prepare the dataset

- Option One: Download the source data and pre-process it locally.

- Option Two: Download the pre-processed data.

## Quick Example

- run FC4 by the following script:

```bash
python launch.py --cfg FC4_confidence.yaml
```

## Extending Anole

> Custom by yourself

We divide the model structure into four categories: 

- Backbone: Used to extract features, it is generally composed of convolution parts of classical models, such as SqueezeNet, AlexNet, etc.
- Neck: Used for feature processing to enhance the expression ability of networks, such as channel re-weighting module in MDLCC, 
- Head: Used to strategically output prediction, such as confidence-weighted pooling in FC4
- Pipeline: Assemble the above four categories of modules. Most models can share a pipeline, such as (FC4, MDLCC), (CLCC, IGTN), (SIIE, TLCC). 

You can focus on customizing a particular category, and we give an [example](https://github.com/YuxiangTang/Anole/blob/master/anole/model/neck/identity_neck.py) for a custom neck in the following:

```python
import torch.nn as nn

from ..builder import NECK  # Need to import the registry.

__all__ = ['identity_neck']

# Custom by yourself
class IdentityNeck(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityNeck, self).__init__()

    def forward(self, x, **kwargs):
        return x

# Need to register first and you can call it in the config.
@NECK.register_obj
def identity_neck(**kwargs):
    return IdentityNeck(**kwargs)

```

## Some Results

