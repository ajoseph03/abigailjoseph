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
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()

def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))
```

### Pre-processing
The images from the dataset are normalized by dividing with 255. They are then reshaped from 28x28 to a 1D tensor with 784 elements.

**Python Code:**
```python
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255

X = X.reshape(X.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)
```

### Visualizing the Dataset
A few images from the dataset are showcased.

**Python Code:**
```python
for i in range(10):
    plt.imshow(X[i, 0, :, :], cmap='gray')
    plt.title(str(Y[i]))
    plt.show()
```

### Model and Training
A simple neural network model is trained on the MNIST data with the aim to reach a target accuracy of 75%.

**Python Code:**
```python
M = GPU(np.random.rand(10, 784))
y = M@X

batch_size = 64
x = X[:, 0:batch_size]
M = GPU(np.random.rand(10, 784))
y = M@x
y = torch.argmax(y, 0)
torch.sum((y == Y[0:batch_size]))/batch_size

for i in range(1000000):
    y = torch.argmax((M@X), axis=1)
    score = ((y == Y).sum(1)/len(Y))
    s = torch.argsort(score, descending=True)
    M = M[s]
    M[50:100] = 0
    M[0:50] = M[0]
    M[1:] += step*GPU_data(np.random.rand(N-1, 10, 784))
    acc = score[s][0].item()
    if acc > acc_best:
        m_best = M[0]
        acc_best = acc47
        print(i, acc)
```

## Observations
Problem set 1 provides a hands-on approach to understanding basic neural network operations using a popular dataset. The usage of `torch` and its functionalities is evident throughout the notebook.
