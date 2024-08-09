# Coputer-Vision-UGR

Assignment 1

### Exercise 1: Mask Discretization and Gaussian Filtering

In this exercise, we discretize convolution masks and apply them to images using OpenCV functions. The exercise emphasizes efficient computation by utilizing 1D (separable) convolutions.

#### Part A: 1D Discrete Mask Computation

- **Task**: Compute the 1D discrete masks of the Gaussian function and its first and second derivatives (normalized). The function should accept either a sigma value or a mask size.
  
  **Details**: 
  - Use sigma values {1, 2.5, 5} and mask sizes {5, 9, 15}.
  - Display the profile of the masks to verify correctness.

- **Solution**:
  - The discrete 1D masks of the Gaussian function, and its 1st and 2nd derivatives are calculated, leveraging the separability of the Gaussian to create 2D Gaussian functions.

##### Gaussian Function

The Gaussian function is defined as:

\[
\text{GaussFunc}(x, \sigma) = c \cdot e^{-\frac{x^2}{2\sigma^2}}
\]

- `c`: A constant (which we can ignore).
- `x`: The point at which we evaluate the function.
- `sigma`: The standard deviation of the mean, or "scale" of the Gaussian kernel, controls the amount of smoothing.

Gaussian functions remove high-frequency components from the image, acting as a low-pass filter and smoothing the image.

##### First Derivative of Gaussian

The first derivative of the Gaussian function with respect to x:

\[
\text{GaussDeriv1Func}(x, \sigma) = \frac{d}{dx} \left(c \cdot e^{-\frac{x^2}{2\sigma^2}}\right) = -\frac{x}{\sigma^2} \cdot \text{GaussFunc}(x, \sigma)
\]

This derivative helps in edge detection by smoothing in one direction and differentiating in the other, leveraging the separability of the kernel function.

##### Second Derivative of Gaussian

The second derivative of the Gaussian function:

\[
\text{GaussDeriv2Func}(x, \sigma) = \frac{d^2}{dx^2} \left(c \cdot e^{-\frac{x^2}{2\sigma^2}}\right) = \frac{x^2 - \sigma^2}{\sigma^4} \cdot \text{GaussFunc}(x, \sigma)
\]

This derivative allows for more precise edge localization.

##### 1D Gaussian Mask

The `gaussianMask1D` function generates a 1D discrete Gaussian mask based on either the specified `sigma` or the desired `sizeMask`.
  - The mask is centered around zero and spans from `-k` to `k`, where `k` is determined by `3*sigma`.
  - The function provides formulas to calculate `sizeMask` when `sigma` is given and vice versa.

#### Normalization of 1D Gaussian Masks

Normalization of the 1D Gaussian masks depends on the type of kernel (smoothing or differentiating) and is determined by the `order` parameter:

1. **Gaussian Mask = Smoothing (`order = 0`)**:
   - The mask is normalized by dividing each value by the sum of all mask values.

2. **First Derivative in X = Differentiating (`order = 1`)**:
   - The derivative of the Gaussian function is computed and scaled by `sigma` to recover the original amplitude.

3. **Second Derivative in X = Differentiating (`order = 2`)**:
   - The second derivative of the Gaussian function is calculated and scaled by `sigma^2` to avoid shrinking and recover response.

##### Visualization

For this part, the Gaussian function is plotted for two cases:

1. **Sigma given**: Calculate `sizeMask` using `sizeMask = 2 [3 * sigma] + 1`.
2. **SizeMask given**: Calculate `sigma` using `sigma = (T - 1) / 6`.

The silhouettes of the masks are displayed as 1D functions to verify correctness.

#### Part B: Filtering and Convolution

**Task**: 
- Read a grayscale image and filter it using convolutions with a Gaussian, the first derivative of the Gaussian (both in \(X\) and \(Y\)), and the second derivative of the Gaussian (both in \(X\) and \(Y\)).
- Use the following sigmas: \(\{0.75, 2, 5, 10, 15\}\).
- Display all images within the same canvas.

In this section, the OpenCV function `sepFilter2D()` must be used with the masks to perform the filtering efficiently.

**The Solution:** In this part, I provide a theory behind developing the 2D Gaussian and its derivative, using separability property. I create a function that, given an image, a sigma value, and a list of orders (representing derivative orders along rows and columns), provides a new image as the result of the convolution.

### **The Separability**
1. **Concept of Separability:**
   - As I mentioned in the clause 1.A, Gaussian has a property of separability. It means that 2D Gaussian can be expressed as the product of two functions, one a function of x and other a function of y. Here, the two functions are the 1D Gaussian.
   - Mathematically, 2D Gaussian `G(x,y)` can be represented as:
   $$\text{G}(x,y) = G_x(x) \cdot G_y(y)$$

2. **Gaussian Derivatives:**
   - The first derivative of a 2D Gaussian is also separable. This allows us to calculate derivatives by applying 1D kernels.
   - Mathematically, it is represented as:
   $$ \frac{dG(x,y)}{dx} = G'_x(x) \cdot G_y(y)$$
    - `G'x`: A horizontal 1D Gaussian derivative kernel
    - `Gy`: A vertical 1D Gaussian kernel.
    - We smooth in one direction, differentiate in the other.
  - The second derivative of Gaussian is separable as well:
   $$ \frac{d^2G}{dx^2} + \frac{d^2G}{dy^2} = G''_h(x) \cdot G_v(y) + G_h(x) \cdot G''_v(y)$$
    - `G''h(x)*Gv(y)`: Convolution in one direction with the 2nd derivative of Gaussian and then, in the other direction, convolution with Gaussian.
    - `Gh(x)*G''v(y)`: Convolution in one direction with Gaussian and then, in the other direction, convolution with the 2nd derivative of Gaussian.

### **The `my2DConv` Function**
I make use of the separability of filters: first, we convolve in one direction (1D) and then in the other (1D). So here I perform 2D convolutions using the 1D masks created in the previous section.
1. **Implementation:**
   - The inputs:
     - `im`: The input image.
     - `sigma`: The standard deviation of the mean.
     - `orders`: A list of the desired derivative orders (`[0, 0]`, `[1, 0]`, ...).
   - The function:
     - Computes 1D Gaussian `maskG`, 1st derivative `maskDG`, and 2nd derivative `mask2DG`, using the `gaussianMask1D` function, developed in the previous step.
     - Applies different operations, depending on the orders:
       - For `[0, 0]`, it performs Gaussian smoothing using 1D Gaussian kernels in both directions.
       - For `[1, 0]`, it applies the first derivative in the X direction.
       - For `[0, 1]`, it applies the first derivative in the Y direction.
       - For `[2, 0]` and `[0, 2]`, it applies the second derivative in the X and Y directions, respectively.
    - As a summary: I break down the 2D convolution into two 1D convolutions, by using `gaussianMask1D` and OpenCV function `sepFilter2D()`.

### **Checking the function**
1. **Image:**
   - To check the function, I use the image 'zebra.jpg'. It is converted to float to avoid precision problems and truncation of integer values.

2. **Results:**
   - I visualize the outcomes for various orders (`[0, 1]`, `[1, 0]`, `[2, 0]`, `[0, 2]`) and different sigma values.
   - The `cv2.sepFilter2D` function convolves the input image with two 1D kernels, leveraging separability.

3. **Visualization:**
   - I use `ddepth=cv2.CV_64F` instead of `-1` in `sepFilter2D()` for more appropriate visualization.
   - If the input image is uint8, preserves negative values
(negative slopes on black and positive on white).


#### Part C: Laplacian of Gaussian

**The Task:** Use the OpenCV function `sepFilter2D()` and $\sigma=3.0$ to calculate the gradient (4 images: $X$ derivative, $Y$ derivative, magnitude and orientation) and the Laplacian of Gaussian.

**The Solution:** Here, I explain the theory about computing the LoG, gradient of an image, magnitude and orientation using OpenCV functions, particularly `sepFilter2D()`.

### **Gradient Calculation**

The gradient of an image can be computed two ways:
1. By using the Sobel operator, which calculates the first derivatives of the image in the X and Y directions.
2. By using Gaussian masks, with their first and second derivatives.

The magnitude and orientation of the gradient are derived from these derivatives. In this case, I will use second method and detect edges with Gaussian masks.

**Steps:**
- **First Derivative in X Direction (Gx):**
  I use my2DConv function with sigma of 3.0 and orders parameter [1, 0], to compute the first derivative in the X direction. These operation correspond to computing the gradient in the horizontal direction, which is similar to what the Sobel operator does.
- **First Derivative in Y Direction (Gy):**
  Similar to Gx, but computes the first derivative in the Y direction.
- **Gradient Magnitude (Gmag):**
  Calculated as the square root of the sum of squares of Gx and Gy, representing the overall intensity change at each pixel.
- **Gradient Orientation (Gdir):**
  Calculated using the arctan2 function to determine the angle of the gradient vector with respect to the horizontal axis.

The script employs separable convolution operations to efficiently compute the derivatives. Gaussian smoothing is implemented to reduce noise before derivative computation and increase the accuracy of gradient estimation.

### **Laplacian of Gaussian (LoG)**

LoG is useful for edge detection and feature extraction. It is calculated as the sum of two-dimensional convolutions of the image with a Gaussian kernel and the Laplacian operator. Mathematically, this operation can be expressed as:

   $$ \frac{d^2(I*G)}{dx^2} + \frac{d^2(I*G)}{dy^2} = I * (\frac{d^2G}{dx^2} + \frac{d^2G}{dy^2}) = I * \frac{d^2G}{dx^2} + I * \frac{d^2G}{dy^2}$$

It means that the operation can be broken down into a series of one-dimensional convolutions, making the computation more efficient compared to a single two-dimensional convolution. The Gaussian kernel smooths the image, while the Laplacian highlights intensity variations.

Here, the LoG is calculated by performing two separate convolutions:
1. Convolution by rows with the second derivative of the Gaussian.
2. Convolution by columns with the Gaussian, and then convolving by rows with the second derivative of the Gaussian.

**Steps:**
  - The my2DConv function is applied with a sigma of 3.0 and [2, 0] orders, indicating the computation of the second derivative in the X direction after Gaussian smoothing.
  - The resulting image represents the LoG of the input image.

The LoG is computed by applying separable convolutions with Gaussian smoothing followed by the second derivative computation in the X direction.
The resulting image highlights edges and features no matter of their orientation. As the Gaussian Mask is already scaled by sigma and sigma^2, there is no need to scale results this time.

# **Exercise 2**:  Gaussian and Laplacian pyramids

#### Part A

**The Task:** Build a 4-level Gaussian pyramid using $7\times7$ size Gaussian masks. Using the OpenCV `pyrUp()` and `pyrDown()` functions is not allowed.

**The Solution:** In this part, I will provide details about Gaussian Pyramid Construction and theoretical reasoning behinds each steps.

Sometimes, we need to work with
an image at different resolutions:
- When we don't know the size of the object on the image we're looking for
- When we need to access an image with
different levels of blur.
In this cases, we can use Gaussian Pyramid.

### **Gaussian Pyramid**

The Gaussian pyramid is a multi-scale representation of an image, where each level in the pyramid contains a progressively downsampled version of the original image. This pyramid is constructed by applying Gaussian blur and downsampling operations iteratively.

**Steps:**
  
  For each level in the pyramid, except the original image, I'm performing the following steps:

  - **Gaussian Blur:**
      - First, I apply the Gaussian blur to the previous level using a 2D convolution operation. The sizeMask default size is 7x7.
      - The sigma for Gaussian mask is calculated based on the mask size T with the formula, mentioned before:
$$\sigma = \frac{T - 1}{6}$$
  - **Subsampling:**
      - Subsample the blurred image by downsampling it by a factor of 2, creating a lower-resolution version. Here, I tried 2 different approaches, one with cv2.resize and another with straightforward downsampling and experimentally chose second one.
      - Append the subsampled image to the Gaussian pyramid.

After this steps, I return the completed Gaussian pyramid, containing multiple levels of progressively downsampled images.

#### Part B

**The Task:** Use your code from the previous section to implement a function that allows you to build a 4-level Laplacian pyramid.

**The Solution:** Here I explain the steps performed to build 4-level Laplacian pyramid and the theory behind it.

### **Building a Laplacian Pyramid**

The Laplacian pyramid is a multi-scale image representation that captures the details, high-frequency components, of an image at different resolutions. It is constructed by calculating a series of levels, each containing information about the difference between levels of a Gaussian pyramid.

- **Gaussian Pyramid Construction:**
Gaussian smoothing and downsampling operations help in reducing noise and retaining essential image features at different resolutions.
   - The Gaussian pyramid is constructed using the `pyramidGauss` function, developed before. It is built by repeatedly applying Gaussian smoothing and downsampling operations, resulting in a series of images with progressively reduced resolution.

- **Laplacian Computation:**
The Laplacian operation highlights the differences or details between successive levels of the Gaussian pyramid, providing a compact representation of the image's structure.
   - The Laplacian pyramid is computed as the difference between each level of the Gaussian pyramid and the upsampled version of the next level.
   - This difference is calculated using the `cv2.subtract` function, which subtracts corresponding pixel values from the current level and the upsampled next level.
   - The resulting images in the Laplacian pyramid represent the high-frequency components of the image at different resolutions. These details can be used for reconstructing the original image.

- **Upsampling and Subtraction:**
   - To compute the Laplacian at each level, the next level of the Gaussian pyramid is upsampled to match the size of the current level with the `cv2.resize` function.
   - The upsampled image is then subtracted from the current level of the Gaussian pyramid, resulting in the Laplacian image for that level.

- **Final Laplacian Pyramid:**
   - The Laplacian images obtained from each level, are stored in a list to form the Laplacian pyramid, except the last level of the Gaussian pyramid.
   - The last level of the Gaussian pyramid is directly appended to the Laplacian pyramid, as there is no next level to compute its Laplacian.

#### Part C

**The Task:** Now implement a function to reconstruct the image encoded/stored in the Laplacian pyramid. Verify that it is possible to obtain an exact copy of the original image. To do this, calculate the reconstruction error as the Euclidean norm of the differences between the intensity levels of the original image and the reconstructed image.

**The Solution:** In this section, I describe steps performed for image reconstruction with Laplacian pyramid.

### **Reconstructing an Image from a Laplacian Pyramid**

To reconstruct an image from its Laplacian pyramid, we need to expand each level of the pyramid to match the size of the next level and then add it to the corresponding level of the pyramid.

**Steps:**

- **Reconstruction:**
   - Starting from the finest level (the last level of the Laplacian pyramid), it iterates through each level in reverse order, except for the last level.
   - For each level, the Laplacian image is expanded to match the size of the next level with the `cv2.resize` function, the interpolation method is `flagInterp=cv2.INTER_LINEAR`.
   - It adds the expanded image to the Laplacian image of the next level with the `cv2.add` function. As a result we get the reconstructed image for that level.
   - Then it appends the reconstructed image for each level to the `reconstruction_steps` list.
   - At the end, the function returns the final reconstructed image and the list of reconstruction steps.

- **Error Calculation:**
   - After the reconstruction process, the function calculates the reconstruction error. It is computed as the Euclidean norm of the differences between the intensity levels of the original image `im` and the final reconstructed image `reconstructed_image`.

 
# **Exercise 3: Hybrid Images**

This exercise is inspired on the following paper: $\textit{Oliva, A., Torralba, A., & Schyns, P. G. (2006). Hybrid images. ACM Transactions on Graphics (TOG), 25(3), 527-532}$. (https://stanford.edu/class/ee367/reading/OlivaTorralb_Hybrid_Siggraph06.pdf).  

**The Task:** The goal is to learn how the distance affects the human visual system when it extracts information about an object. To do this, we build a hybrid image from two images of different objects. By properly mixing part of the high frequencies of one image with part of the low frequencies of another image, we obtain a hybrid image whose perception differs with distance. The sigma used to filter both images (both high and low frequencies) is the key aspect to select the high and low frequency range of each image. The higher the sigma value, the greater the removal of high frequencies from the image. It is recommended to choose this value separately for each of the images and, in fact, it is possible (and even desirable) to have different values for different pairs of images. Remember to use the Gaussian pyramid to show the effect obtained.

In particular, the students are required to generate, at least, the hybrid images corresponding to Einstein-Marilyn, Cat-Dog, and Fish-Submarine, as shown below.

**The Solution:** In this section, I explain the foundaiton behind generation of hybrid images, which blend the high frequencies of one image with the low frequencies of another, to get an image that looks differently depending on the viewing distance.

### **Hybrid Image Generation**

- **High and Low Frequencies:** Images can be decomposed into high and low frequency components. High frequencies  are the ones with fine details and most accentuated edges in the image. Low frequencies, on the other hand, represent slow changes and correspond to the overall structure and general features of the image, they have a softer appearance.

- **Gaussian Filtering:** To get low frequencies we can use Gaussian masks which filter images. With different sigma values we can control the amount of smoothing applied. The higher the sigma value, the greater the
removal of high frequencies in the convolved
image. To get high frequencies, we can calculate the difference between the original and its smoothed version. By adjusting sigma, we can filter out high and low frequencies from the images. Sometimes it's better to choose sigma values for each of the two image separately.

- **Hybrid Image Generation:** We can create hybrid images by combining the high frequencies of one image with the low frequencies of another. It can be achieved by subtracting the low-pass filtered version of one image from the original image and adding it to the high-pass filtered version of the other image.

**Steps:**

- **Load Images:** Images for the hybrid image generation should consist objects that can be percieved a bit differently at varying distances.

- **Set Sigma Values:** Sigma values determine the amount of smoothing applied to the images and control the frequency content of each image. They are chosen to balance between high-frequency details and the emphasis on low-frequency components in each image pair. Lower sigma values (like 3, 4) are selected for images with intricate details, such as Einstein and Fish, to retain high-frequency information. Higher sigma values (like 8, 10) are chosen for images with smoother features, like Cat and Submarine, to emphasize low-frequency components. For Marylin I dicreased sigma value to balance Einstein detailing.

- **Generate Hybrid Images:** For each image pair, I use the `generateHybridImages` function to create a hybrid image. This function filters each image to extract high and low frequencies and then combines them to create the hybrid image:
  - Calculates Gaussian masks for each image based on the provided sigma values.
  - Smooths the images using the `sepFilter2D` function to extract low-frequency components.
  - Calculates high-frequency components by subtracting the low-pass filtered images from the original images.
  - Combines the low and high frequencies to create the hybrid image.

4. **Display Hybrid Images:** After applying calculated function to loaded images with setted sigma values, to check if the effect is achieved, I built the Gaussian pyramid with the hybrid images. In this case there is no need to move away from the computer to see the effect, which means that the task was solved successfully.

# **Exercise 4: Pyramid Blending**

**Tha Task:** In this exercise you must create the merged image (horizontal) of the apple (`apple.jpg`) and the orange (`orange.jpg`) using the $\textit{Pyramid Blending}$ technique. It is recommended to use the OpenCV functions `pyrDown` and `pyrUp` (and explain its working/functioning). It is also essential to explain in detail the process followed when creating the new image (mixture of both), and the blending effect in the resulting image should be appropriate (like in the example displayed below).  

**The Solution:** Here I explain the steps performed for Pyramid Blending, including decomposing the images into Gaussian and Laplacian pyramids, blending corresponding levels of the Laplacian pyramids, and then reconstructing the final blended image.

### **Pyramid Blending**

- **Start:**
   - First, I converted images to floating-point type for precision.
   - Next, I initialized Gaussian pyramids for both the apple and orange images. These pyramids will help us decompose the images into different scales (levels of detail).

- **Gaussian Pyramids:**
   - To create the Gaussian pyramids, I'm downsampling image at each level by reducing its resolution by half with `pyrDown` function. This process captures the low-frequency components, blurry details, of the images.

- **Laplacian Pyramids:**
   - I initialize the Laplacian pyramids using the last level of the Gaussian pyramids. The highest level of the Laplacian pyramid corresponds to the original image.
   - For each level of the Gaussian pyramids, I'm performing the following steps:
       - Upsample the image to match the dimensions of the next lower level with the `pyrUp` function.
       - Subtract the upsampled version from the original level to obtain the Laplacian pyramid.
       - The Laplacian pyramid represents the high-frequency details (edges, textures) of the images.
   - This decomposition simplifies blending and isolates fine details.

- **Blending:**
   - To blend corresponding levels of the Laplacian pyramids I use horizontal mask. It plays a crucial role in determining how much of each image contributes to the final result. I assigned the upper part of the array with the size of the images to 1 and bottom part to 0.
   - Then element-wise multiplication is performed between the Laplacian coefficients and the mask, and added to the corresponding coefficients of the other image multiplied by (1 - mask).

- **Reconstructing the Blended Image:**
   - I'm obtaining the final blended image by reconstructing the blended Laplacian pyramid.
   - For this I upsample each level of the blended Laplacian pyramid to match the dimensions of the next higher level.
   - Then adding the upsampled coefficients to the corresponding coefficients of the next lower level gives to get the reconstructed image.
   - This way I obtain the final image, which contains both the low-frequency components from the Gaussian pyramid and the high-frequency details from the Laplacian pyramid.

- **Displaying results**
   - To ensure valid pixel values for display, we clip the resulting pixel values to the range [0, 255].
   - The final result is converted back to 8-bit format (uint8) for visualization.

