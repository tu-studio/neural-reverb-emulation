# Neural Reverb Emulation

## Setup Guide

### 1. Create a virtual environment

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

Or using conda:
```bash
# Create conda environment
conda create -n reverb-env python=3.8

# Activate environment
conda activate reverb-env
```

### 2. Install Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

### 3. Main Dependencies
- PyTorch
- torchaudio
- tensorboard
- tqdm
- numpy
- pyyaml

### 4. GPU Support
- CUDA-capable GPU recommended for training
- Update `device_request` in `params.yaml` to use GPU (`cuda`) or CPU (`cpu`)

### 5. Running the Project

After setting up the environment and installing requirements:
1. Update configuration in `params.yaml`
2. Run training:
```bash
python train.py
```

For monitoring training:
```bash
tensorboard --logdir logs/tensorboard
```