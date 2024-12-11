# Neural Reverb Emulation

This repository aims to emulate the sound of a plate reverb EMT 140 using deep learning. The project is fully integrated into a DVC pipeline designed for the HPC cluster at TU.


## Setup Guide

After cloning the repository on the cluster, follow those steps:

### 1. Create a Virtual Environment and Install Requirements

**Using `venv` (recommended):**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Unix or macOS:
source venv/bin/activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```


### 2. Run the Project


Training is initiated through the `./exp` script, with parameters configured in `params.yaml`, and steps configured in `dvc.yaml`. Progress and results are logged to TensorBoard.

**On the HPC cluster:**

```bash
./exp_workflow.sh
```

### 3. Monitoring Training

To monitor progress on the HPC cluster:

```bash
source venv/bin/activate
tensorboard --logdir=~/Data/neural-reverb-emulation/logs/tensorboard --path_prefix=/tb1 &!
```


## Repository Organization

- **Model code**: Located in `source/network`
- **Pipeline execution**: Managed by the `dvc.yaml` script


## Parameters

Our architecture if fully modulable, and most parameters of it can be tweaked to research the best model.

### Training parameters

| Parameter       | Description                                       |
|-----------------|---------------------------------------------------|
| `input_file`    | Path to preprocessed training data                |
| `n_bands`       | Number of frequency bands for PQMF (1 = disabled) |
| `n_blocks`      | Number of TCN blocks in encoder/decoder          |
| `kernel_size`   | Size of convolutional kernels                    |
| `dilation_growth` | Factor for dilation growth between blocks        |
| `dilate_conv`   | Use dilated convolutions in latent space         |
| `n_channels`    | Base number of channels (doubles with each block) |
| `stride`        | Stride of convolutions                           |
| `lr`            | Learning rate                                    |
| `use_kl`        | Enable VAE with KL divergence                    |
| `use_skip`      | Enable skip connections                          |
| `use_latent`    | Latent space type (`dense` or `conv`)            |
| `use_tcn`       | Use TCN instead of autoencoder                  |
| `use_pqmf`      | Enable PQMF analysis/synthesis                  |
| `use_adversarial` | Enable adversarial training phase                |
| `use_noise`     | Add learned noise to decoder output             |
| `use_wn`        | Enable weight normalization                     |
| `use_batch_norm` | Enable batch normalization                       |
| `use_residual`  | Enable residual connections in blocks            |
| `use_upsampling` | Use transposed convolutions vs regular           |
| `activation`    | Activation function (`prelu` or `leaky_relu`)    |
| `padding`       | Amount of zero padding                          |


### Metrics Parameters

| Parameter           | Description                                      |
|---------------------|--------------------------------------------------|
| `loss_function`     | Main objective (`mse`, `spectral_distance`, etc.)|
| `additional_mse`    | Track additional MSE metric                     |
| `additional_spec`   | Track additional spectral distance metric       |
| `additional_stft`   | Track additional STFT loss                      |
| `additional_fft`    | Track additional FFT loss                       |


### Discriminator Parameters (Adversarial Training)

| Parameter     | Description                     |
|---------------|---------------------------------|
| `lr`          | Discriminator learning rate     |
| `beta1`       | Adam optimizer beta1 parameter  |
| `beta2`       | Adam optimizer beta2 parameter  |
| `n_layers`    | Number of discriminator layers  |
| `n_channels`  | Channels per layer             |
| `kernel_size` | Size of convolutional kernels   |
| `stride`      | Stride of convolutions          |
| `padding`     | Amount of zero padding          |
| `loss_type` | Adversarial loss function (`hinge`, `square`) |


### General Parameters

| Parameter      | Description                     |
|----------------|---------------------------------|
| `sample_rate`  | Audio sample rate              |
| `random_seed`  | Seed for reproducibility        |
| `input_size`   | Size of input audio chunks      |



## Key Scripts

### CompressionRate.py

This script identifies optimal model configurations by:
- Calculating receptive field sizes for various architectures
- Performing grid searches across parameters (e.g., blocks, kernel size, dilation)
- Optimizing for target receptive field sizes
- Exploring configurations for different frequency bands

### Multi_submission.py

This script facilitates large-scale training by:
- Generating multiple hyperparameter combinations
- Automating SLURM batch job submissions for different configurations
- Supporting experiments with diverse architectures (e.g., VAE, adversarial training, skip connections)
- Handling parameter sweeps like latent dimensions, kernel sizes, and block counts



