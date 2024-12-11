# Neural Reverb Emulation

This repository aims at emulating the sound of a plate reverb EMT 140 using deep learning.

It is fully integrated in a DVC pipeline for the hpc cluster of TU.

## Setup Guide

### 1. Create a virtual environment and install requirements

Using venv (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Unix or MacOS:
source venv/bin/activate
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 2 . Repository organization

All the relevant code for the model is in source/network

The project runs following the dvc.yaml script.

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| input_file | Path to preprocessed training data |
| n_bands | Number of frequency bands for PQMF analysis/synthesis (1 = disabled) |
| n_blocks | Number of TCN blocks in encoder/decoder |
| kernel_size | Size of convolutional kernels |
| dilation_growth | Multiplicative factor for dilation between blocks |
| dilate_conv | Whether to use dilated convolutions in latent space |
| n_channels | Base number of channels (doubles with each block) |
| stride | Stride of convolutions |
| lr | Learning rate |
| use_kl | Enable variational autoencoder with KL divergence |
| use_skip | Enable skip connections between encoder-decoder |
| use_latent | Latent space type ('dense'=linear layer, 'conv'=convolutional) |
| use_tcn | Use simple TCN instead of autoencoder |
| use_pqmf | Enable pseudo-QMF filterbank analysis/synthesis |
| use_adversarial | Enable adversarial training phase |
| use_noise | Add learned noise to decoder output |
| use_wn | Enable weight normalization |
| use_batch_norm | Enable batch normalization |
| use_residual | Enable residual connections in blocks |
| use_upsampling | Use transposed convolutions vs regular |
| activation | Activation function type ('prelu' or 'leaky_relu') |
| padding | Amount of zero padding |

## Metrics Parameters

| Parameter | Description |
|-----------|-------------|
| loss_function | Main training objective (Options: 'mse', 'spectral_distance', 'single_stft_loss', 'fft_loss', 'combined') |
| additional_mse | Enable additional MSE metric tracking |
| additional_spec | Enable additional spectral distance metric tracking |
| additional_stft | Enable additional STFT loss metric tracking |
| additional_fft | Enable additional FFT loss metric tracking |

## Discriminator Parameters

Used when adversarial training is enabled:

| Parameter | Description |
|-----------|-------------|
| lr | Discriminator learning rate |
| beta1 | Adam optimizer beta1 parameter |
| beta2 | Adam optimizer beta2 parameter |
| n_layers | Number of discriminator layers |
| n_channels | Channels per layer |
| kernel_size | Size of convolutional kernels |
| stride | Stride of convolutions |
| padding | Amount of zero padding |

## GAN Parameters

| Parameter | Description |
|-----------|-------------|
| loss_type | Adversarial loss function (Options: 'hinge', 'square') |

## General Parameters

| Parameter | Description |
|-----------|-------------|
| sample_rate | Audio sample rate |
| random_seed | Seed for reproducibility |
| input_size | Size of input audio chunks |

## Usage

Parameters can be configured in the `params.yaml` file. Example:

```yaml
train:
  n_bands: 1
  n_blocks: 4
  kernel_size: 16
  # ... other parameters

metrics:
  loss_function: 'combined'
  additional_mse: true
  # ... other metrics settings
```


### 3 . Running the Project

On the hpc cluster: 
```bash
./exp_workflow.sh
```

Training can be initiated using the `./exp` script, with parameters configured through `params.yaml`. Training progress and results are logged to TensorBoard.

## Script Descriptions

### CompressionRate.py
This script helps find optimal model configurations by:
- Calculating receptive field sizes for different architectures
- Performing grid search across different parameters (blocks, kernel size, dilation)
- Finding best configurations for different numbers of frequency bands
- Optimizing the architecture to achieve target receptive field sizes

### Multi_submission.py
This script automates large-scale model training by:
- Generating multiple hyperparameter combinations
- Finding optimal parameters for different frequency band configurations
- Submitting batch jobs to a SLURM cluster with different configurations
- Supports experimentation with various model architectures (VAE, adversarial training, skip connections, etc.)
- Handles parameter combinations like latent dimensions, kernel sizes, number of blocks, etc.
- Uses environment variables to pass parameters to SLURM jobs

## Usage

1. Configure your parameters in `params.yaml`
2. Run CompressionRate.py to find optimal architecture configurations
3. Use Multi_submission.py to launch multiple training jobs with different parameters
4. Monitor training progress through TensorBoard logs

### 4 . Monitor trainings

on marckh:

```bash
source venv/bin/activate
tensorboard --logdir=~/Data/neural-reverb-emulation/logs/tensorboard  --path_prefix=/tb1 &!
```

