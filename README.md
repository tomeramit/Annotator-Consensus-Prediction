This is the official repository of the paper [Annotator Consensus Prediction for Medical Image Segmentation with Diffusion Models](https://arxiv.org/abs/2306.09004)

The code is based on [Improved Denoising Diffusion Probabilistic Models](https://github.com/openai/improved-diffusion) and [SegDiff: Image Segmentation with Diffusion Probabilistic Models.](https://github.com/tomeramit/SegDiff)

## Installation
### Conda environment
To create the environment use the conda environment command
```
conda env create -f environment.yml
```

## Project structure and data preparations
Our project need to be arranged in the following format

```
Annotator-Consensus-Prediction/ # git clone the source code here

data/ # the root of the data folders
    brain_growth/
    brain_tumor/
    kidney/
    prostate/
```

All datasets were taken from [Quantification of Uncertainties in Biomedical Image Quantification Challenge (QUBIQ) 2021](https://qubiq.grand-challenge.org/).
Download the datasets from the following [link](https://qubiq21.grand-challenge.org/participation/).

The datasets should have the following format
```
<dataset_name>/
    train/*
    val/*
```

## Train and Evaluate
Execute the following commands (multi gpu is supported for training, set the gpus with CUDA_VISIBLE_DEVICES and -n for the actual number)

Most important training options:
```
# Training
--batch-size    Batch size
--lr            Learning rate

# Architecture
--rrdb_blocks       Number of rrdb blocks
--dropout           Dropout
--diffusion_steps   Number of steps for the diffusion model

# Ablation variants (described in the paper)
--no_annotator_training
--annotators_training
--soft_label_training
--consensus_training (ours)

# Misc
--save_interval     interval for saving model weights
```

Training script example (different script for each dataset):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 mpiexec -n 4 python image_train_diff_brain_tumor_1.py --save_interval 2500 --batch_size 1 --lr 0.00002 --diffusion_steps 100 --no_annotator_training True
```

Evaluation script example (different script for each dataset):
```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python image_sample_diff_brain_tumor_1.py --model_path <path-for-model-weights>
```
