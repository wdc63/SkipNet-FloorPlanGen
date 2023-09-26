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

[Dataset Link](https://o365skku-my.sharepoint.com/:u:/g/personal/jyt0131_o365_skku_edu/Ed6FNmZZm5xDrlAuw7jtRO8BABKiPcMk1d_1oFZtqPCaTA?e=kOWope)

## Training

Run the following command for training:
```bash
python train.py
```

We provide our training model's weight : [model_weight](https://o365skku-my.sharepoint.com/:u:/g/personal/jyt0131_o365_skku_edu/ESUjetq2F-hAiFL9GxsU2dMBK9yyCl43UhhcCfdZD0d0PQ?e=g61IfC)

## Evaluation and Submission

Run the following command for evaluation and submission:
```bash
python submission.py
```

## Inference Visualization

Run the inference_visualization.ipynb Jupyter Notebook for inference visualization.

