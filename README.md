# SkipNet-FloorPlanGen

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preprocessing](#dataset-preprocessing)
3. [Training](#training)
4. [Evaluation and Submission](#evaluation-and-submission)
5. [Inference Visualization](#inference-visualization)

## Environment Setup

Create a Conda environment:
```bash
conda create -n plangen python==3.9.0
conda activate plangen
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Preprocessing

We utilize mobileSAM to preprocess the boundary images and organize the dataset into train/val/test sets for training.

Below is the link to the actual preprocessed dataset used for training:

[Dataset Link](#)

## Training

Run the following command for training:
```bash
python train.py
```

## Evaluation and Submission

Run the following command for evaluation and submission:
```bash
python submission.py
```

## Inference Visualization

Run the inference_visualization.ipynb Jupyter Notebook for inference visualization.

