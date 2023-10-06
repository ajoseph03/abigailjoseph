# MNIST Colab Notebook Summary

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Content Overview](#content-overview)
   - [Loading the Dataset](#loading-the-dataset)
   - [Helper Functions](#helper-functions)
   - [Pre-processing](#pre-processing)
   - [Visualizing the Dataset](#visualizing-the-dataset)
   - [Model and Training](#model-and-training)
4. [Observations](#observations)

## Introduction
This notebook demonstrates loading, visualizing, and running a simple neural network model on the MNIST dataset using PyTorch and related libraries. The model aims to reach a target accuracy of 75% during its training process.

## Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread
```

## Content Overview

### Loading the Dataset
The dataset choices provided are:
- MNIST
- KMNIST (commented out)
- Fashion MNIST (commented out)

**Python Code:**
```python
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)
```

### Helper Functions
These functions help move data to the GPU, visualize data, and plot montages.

**Python Code:**
```python
def GPU(data):
    ...
def GPU_data(data):
    ...
def plot(x):
    ...
def montage_plot(x):
    ...
```

### Pre-processing
The images from the dataset are normalized by dividing with 255. They are then reshaped from 28x28 to a 1D tensor with 784 elements.

**Python Code:**
```python
X = train_set.data.numpy()
...
Y_test = GPU_data(Y_test)
```

### Visualizing the Dataset
A few images from the dataset are showcased.

**Python Code:**
```python
for i in range(10):
    ...
    plt.show()
```

### Model and Training
A simple neural network model is trained on the MNIST data with the aim to reach a target accuracy of 75%.

**Python Code:**
```python
M = GPU(np.random.rand(10, 784))
...
print(i, acc)
```

## Observations
This notebook provides a hands-on approach to understanding basic neural network operations using a popular dataset. The usage of `torch` and its functionalities is evident throughout the notebook.

---

The Table of Contents provides direct links to each section, which can be useful for long README files or for readers who want to jump to a specific section directly.
