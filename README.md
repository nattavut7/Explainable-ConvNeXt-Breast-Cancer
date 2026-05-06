# Explainable ConvNeXt Framework for Breast Cancer Classification

This repository provides the implementation for the paper:

**Explainable ConvNeXt Framework with Attention Rollout for Breast Cancer Classification from Histopathological Images**

## Features
- ConvNeXt backbone
- Test-Time Augmentation (TTA)
- Attention Rollout explainability
- BreakHis dataset support
- PyTorch implementation

## Installation

pip install -r requirements.txt

## Training

python training/train.py

## Inference with TTA

python training/tta_inference.py

## Explainability

python explainability/attention_rollout.py

## Dataset

BreakHis dataset (Breast Cancer Histopathological Dataset).

## Run in Google Colab

To reproduce the experiments easily, you can run the notebook directly in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/nattavut7/Explainable-ConvNeXt-Breast-Cancer/blob/main/notebooks/BreastCancer_ConvNeXt_Colab.ipynb)

Steps:

1. Open the notebook in Colab
2. Install required packages
3. Download the BreakHis dataset
4. Train the ConvNeXt model
5. Run inference with Test-Time Augmentation
6. Generate attention rollout heatmaps
   
## Repository Structure

- Explainable-ConvNeXt-Breast-Cancer
  - README.md
  - requirements.txt
  - dataset
    - download_breakhis.py
  - models
    - convnext_model.py
  - preprocessing
  - explainability
    - attention_rollout.py
  - training
    - train.py
    - tta_inference.py
  - utils
    - dataset_loader.py
    - metrics.py
  - notebooks
    - BreastCancer_ConvNeXt_Colab.ipynb
  - weights
# Explainable-ConvNeXt-Breast-Cancer

Official repository for the paper:

## Paper
**Explainable ConvNeXt Framework with Attention Rollout for Breast Cancer Classification from Histopathological Images**

Nattavut Sriwiboon

Published in *Discover Artificial Intelligence*, 2026.

DOI: https://doi.org/10.1007/s44163-026-01068-8

Paper Link:
https://link.springer.com/article/10.1007/s44163-026-01068-8

## Abstract
This paper proposes an explainable deep learning framework that integrates a ConvNeXt backbone, Test-Time Augmentation (TTA), and Attention Rollout (AR) for breast cancer classification from histopathological images. The proposed framework has achieved 98.7% classification accuracy on the BreakHis dataset while improving interpretability and computational efficiency.

## Citation
```bibtex
@article{sriwiboon2026convnext,
  title={Explainable ConvNeXt framework with attention rollout for breast cancer classification from histopathological images},
  author={Sriwiboon, Nattavut},
  journal={Discover Artificial Intelligence},
  year={2026},
  doi={10.1007/s44163-026-01068-8}
}

