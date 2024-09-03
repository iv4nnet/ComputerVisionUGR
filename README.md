# Computer Vision Assignments

Welcome to my repository for the Computer Vision course assignments from UGR. This repository contains the implementations and solutions for various exercises focusing on image processing, machine learning, and deep learning techniques using Python, OpenCV, and TensorFlow. Below is a brief overview of the assignments included in this repository.

## Table of Contents

- [Assignment 1](#assignment-1)
  - [Exercise 1: Mask Discretization and Gaussian Filtering](#exercise-1-mask-discretization-and-gaussian-filtering)
  - [Exercise 2: Gaussian and Laplacian Pyramids](#exercise-2-gaussian-and-laplacian-pyramids)
  - [Exercise 3: Hybrid Images](#exercise-3-hybrid-images)
  - [Exercise 4: Pyramid Blending](#exercise-4-pyramid-blending)
- [Assignment 2](#assignment-2)
  - [Exercise 1: BaseNet in CIFAR100](#exercise-1-basenet-in-cifar100-3-points)
    - [Model Description](#model-description)
    - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    - [BaseNet Architecture](#basenet-architecture)
    - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Parameter Count](#parameter-count)
  - [Exercise 2: Improvement of the BaseNet Model](#exercise-2-improvement-of-the-basenet-model-35-points)
    - [Incremental Improvements](#incremental-improvements)
    - [Improved BaseNet #7](#improved-basenet-7)
    - [Conclusions](#conclusions)

## Assignment 1

### Exercise 1: Mask Discretization and Gaussian Filtering

This exercise involves discretizing convolution masks and applying them to images using OpenCV functions. The focus is on efficient computation using separable (1D) convolutions.

#### Part A: 1D Discrete Mask Computation

- **Objective**: Compute the 1D discrete masks of the Gaussian function and its first and second derivatives. The function allows the use of either a sigma value or a mask size.
- **Details**:
  - Test with sigma values: {1, 2.5, 5} and mask sizes: {5, 9, 15}.
  - Display the profile of the computed masks for verification.

#### Part B: Filtering and Convolution

- **Objective**: Apply convolutions to a grayscale image using Gaussian and its derivatives.
- **Details**:
  - Use the following sigma values: {0.75, 2, 5, 10, 15}.
  - Apply convolutions in both X and Y directions.
  - Display the results on the same canvas for comparison.

#### Part C: Laplacian of Gaussian

- **Objective**: Calculate the gradient and Laplacian of Gaussian using the `sepFilter2D()` function with \(\sigma = 3.0\).
- **Details**:
  - Compute gradients in X and Y directions, magnitude, and orientation.
  - The resulting image should highlight edges regardless of their orientation.

### Exercise 2: Gaussian and Laplacian Pyramids

#### Part A: Gaussian Pyramid

- **Objective**: Build a 4-level Gaussian pyramid using \(7 \times 7\) Gaussian masks.
- **Details**: Do not use the OpenCV `pyrUp()` and `pyrDown()` functions.

#### Part B: Laplacian Pyramid

- **Objective**: Build a 4-level Laplacian pyramid using the Gaussian pyramid.
  
#### Part C: Image Reconstruction

- **Objective**: Reconstruct the original image from the Laplacian pyramid.
- **Details**: Verify the accuracy of reconstruction by calculating the Euclidean norm of the difference between the original and reconstructed images.

### Exercise 3: Hybrid Images

Inspired by the paper "Hybrid Images" by Oliva et al. (2006), this exercise explores how distance affects the human visual system when perceiving an image. 

- **Objective**: Create a hybrid image by blending high-frequency components of one image with low-frequency components of another.
- **Details**:
  - Use Gaussian pyramids to control the frequency content of each image.
  - Experiment with different sigma values for filtering.

### Exercise 4: Pyramid Blending

This exercise focuses on creating a blended image by merging two images (e.g., an apple and an orange) using the Pyramid Blending technique.

- **Objective**: Create a horizontally merged image using pyramid blending.
- **Details**:
  - Use OpenCV’s `pyrDown` and `pyrUp` functions.
  - Explain the process and the resulting blending effect in detail.

## Assignment 2: Deep Learning for Computer Vision

The goal of this assignment is to gain hands-on experience in designing and training deep neural networks, particularly CNNs, using the fastai library. Starting from a baseline model, `BaseNet`, this assignment explores how to modify, enhance, and optimize a neural network for image classification tasks using part of the CIFAR-100 dataset.

### Exercise 1: BaseNet in CIFAR100 (3 points)

#### Model Description

In this exercise, we work with a subset of the CIFAR-100 dataset, focusing on 25 classes. The goal is to create a baseline CNN model, `BaseNet`, and evaluate its performance in classifying images from these classes.

#### Data Loading and Preprocessing

- **Dataset**: CIFAR-100 subset with 25 classes.
- **Training Data**: 12,500 images.
- **Validation Data**: 2,500 images (10% of the training set).
- **Data Preprocessing**:
  - Labels extracted from folder names.
  - Data augmentation applied during training (e.g., random zoom, rotation, and horizontal flip) to increase dataset diversity.

#### BaseNet Architecture

The `BaseNet` architecture is defined as follows:

1. **Convolutional Layer 1**:
   - Input: 3 channels (RGB).
   - Output: 4 channels.
   - Kernel size: 7x7.

2. **ReLU Activation**:
   - Introduces non-linearity.

3. **Max Pooling**:
   - Reduces spatial dimensions.

4. **Convolutional Layer 2**:
   - Input: 4 channels.
   - Output: 10 channels.
   - Kernel size: 5x5.

5. **ReLU Activation**:
   - Introduces non-linearity.

6. **Flatten Layer**:
   - Converts output to a 1D tensor.

7. **Fully Connected Layer 1**:
   - Input: 810 features.
   - Output: 50 features.

8. **ReLU Activation**:
   - Introduces non-linearity.

9. **Fully Connected Layer 2**:
   - Output: 25 classes (multiclass classification).

#### Model Training and Evaluation

- **Training**:
  - Used fastai's `Learner` object, combining the dataset, `BaseNet` model, cross-entropy loss, and accuracy metric.
  - Trained using the `fit` method.
- **Evaluation**:
  - Accuracy and loss metrics on the validation set.
  - Confusion matrix analysis to assess class-wise performance.

#### Parameter Count

- Total number of parameters in the `BaseNet` model: _[Calculation needed]_
- Breakdown of the parameter count for each layer and component.

### Exercise 2: Improvement of the BaseNet Model (3.5 points)

#### Incremental Improvements

The goal of this exercise is to enhance the `BaseNet` model by making judicious architectural and implementation choices. The enhancements include:

- **Data Normalization**:
  - Used DataBlock class to normalize input data (mean=0, stddev=1).
- **Batch Normalization**:
  - Added after convolutional layers to reduce overfitting.
- **Depth Increase**:
  - Added more convolutional layers without excessive max-pooling.
- **Dropout Regularization**:
  - Introduced dropout to prevent overfitting.
- **Early Stopping**:
  - Monitored training and validation curves to avoid overfitting.
- **Data Augmentation**:
  - Increased augmentation with `aug_transforms(mult=2.0)`.

#### Improved BaseNet #7

The final fine-tuned version of `BaseNet`, named `Improved BaseNet #7`, achieved a significant accuracy improvement:

- **Model Complexity**:
  - Increased depth with more convolutional layers (64 and 128 filters).
- **Hyperparameter Tuning**:
  - Used default learning rate and batch size, but highlighted the potential for future tuning.
- **Data Augmentation**:
  - Aggressive augmentation with `aug_transforms(mult=2.0)` to enhance generalization.
- **Dropout Regularization**:
  - Applied dropout rates of 0.25 and 0.5.
- **Early Stopping**:
  - Trained for 10 epochs based on validation loss and accuracy.

#### Conclusions

- **Depth vs. Computational Cost**:
  - Deeper networks can improve accuracy but require more computational resources.
- **Hyperparameter Sensitivity**:
  - Learning rate and batch size are critical; further tuning may yield better results.
- **Data Augmentation Trade-Off**:
  - Aggressive augmentation can improve robustness but may introduce noise.
- **Model Complexity vs. Overfitting**:
  - Regularization techniques like dropout are essential to avoid overfitting.

