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
(https://colab.research.google.com/github/nattavut/Explainable-ConvNeXt-Breast-Cancer/blob/main/notebooks/BreastCancer_ConvNeXt_Colab.ipynb)

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
