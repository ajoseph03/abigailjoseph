# MNIST Colab Notebook Summary

## Introduction
This notebook demonstrates loading, visualizing, and running a simple neural network model on the MNIST dataset using PyTorch and related libraries. The model aims to reach a target accuracy of 75% during its training process.

## Dependencies
- numpy
- matplotlib
- torch
- torchvision
- skimage
- wandb (Weights and Biases for logging and monitoring)

## Content Overview

### Loading the Dataset
The dataset choices provided are:
- MNIST
- KMNIST (commented out)
- Fashion MNIST (commented out)

### Helper Functions
- `GPU(data)`: Moves the given data to the GPU and sets it up for gradient computation.
- `GPU_data(data)`: Moves the given data to the GPU without enabling gradient computation.
- `plot(x)`: Plots the given image data.
- `montage_plot(x)`: Plots a montage of the provided images.

### Pre-processing
The images from the dataset are normalized by dividing with 255. The images are then reshaped from 28x28 to a 1D tensor with 784 elements.

### Visualizing the Dataset
The first 10 images from the training set are displayed.

### Model and Training
1. Random initialization of model weights.
2. The model is trained in a loop, aiming to reach a target accuracy of 75%.
3. The training process involves updating weights and checking accuracy. The best weights with the highest accuracy are stored.

## Observations
This notebook provides a hands-on approach to understanding basic neural network operations using a popular dataset. The usage of `torch` and its functionalities is evident throughout the notebook.
