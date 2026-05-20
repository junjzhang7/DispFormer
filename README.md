# DispFormer: A Dual Attention Transformer with Denoising for Irregular Time Series Classification

## Overview
This repository is the official implementation of DispFormer: A Dual Attention Transformer with Denoising for Irregular Time Series Classification.

## Requirements
```shell
torch==2.1.0
lightning==2.2.0
wandb==0.17.9
```

## Data Preparation
We adopt the dataset preprocessed by [RainDrop](https://github.com/mims-harvard/Raindrop).

The raw datasets can be found at:

P12: https://physionet.org/content/challenge-2012/1.0.0/

P19: https://physionet.org/content/challenge-2019/1.0.0/

PAM: http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

And the preprocessed dataset can be download from:

P12: https://doi.org/10.6084/m9.figshare.19514341.v1

P19: https://doi.org/10.6084/m9.figshare.19514338.v1

PAM: https://doi.org/10.6084/m9.figshare.19514347.v1

## Run the Code
After obtaining the preprocessed datasets, you can run the model with the following script:
```shell
bash scripts/run.sh
```
For more details, please refer to [./scrips](./scripts/) and [run.py](./src/run.py).


## Acknowledgement
This codebase is constructed based on the repo: [RainDrop](https://github.com/mims-harvard/Raindrop). Thanks a lot for their amazing work!