# Computer Vision Homework Assignments

Welcome to my repository for the Computer Vision course assignments from UGR. This repository contains the implementations and solutions for various exercises focusing on image processing, machine learning, and deep learning techniques using Python, OpenCV, and TensorFlow. Below is a brief overview of the assignments included in this repository.

## Table of Contents

- [Assignment 1](#assignment-1)
  - [Exercise 1: Mask Discretization and Gaussian Filtering](#exercise-1-mask-discretization-and-gaussian-filtering)
  - [Exercise 2: Gaussian and Laplacian Pyramids](#exercise-2-gaussian-and-laplacian-pyramids)
  - [Exercise 3: Hybrid Images](#exercise-3-hybrid-images)
  - [Exercise 4: Pyramid Blending](#exercise-4-pyramid-blending)

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
