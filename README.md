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

### 2 . Repository organization

All the relevant code for the model is in source/network

The project runs following the dvc.yaml script.

All parameters are inside params.yaml


### 3 . Running the Project

On the hpc cluster: 
```bash
./exp_workflow.sh
```

### 4 . Monitor trainings

on marckh:

```bash
source venv/bin/activate
tensorboard --logdir=~/Data/neural-reverb-emulation/logs/tensorboard  --path_prefix=/tb1 &!
```
