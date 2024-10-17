### 1. Semantic Segmentation with U-net

Semantic segmentation is a computer vision algorithm that helps us understand and label every single pixel in an image. It goes beyond just detecting objects or drawing bounding boxes around them.

With semantic segmentation, we can accurately outline the boundaries of objects and determine which pixels belong to each object.

#### 1.1 Per-pixel class label

![per pixel ](https://github.com/user-attachments/assets/421ad7ec-92b0-46a3-8c4c-69327b1103e1)

### 2. Transpose Convolution

- We use a filter, which is like a small window, and we place it on the output grid.
- Then, we multiply each value in the filter with the corresponding value in the input grid. We repeat this process for every position in the output grid.
- We can choose the `size of filter`, `amount of padding`, and `stride`. These parameters affect how the transpose convolution works and the final output size.

![Transpose conv](https://github.com/user-attachments/assets/2a225321-8118-4eec-b60e-83988d0fe4ee)

### 3. U-Net Architecture

- The input to the U-Net is an image, represented as a grid of pixels with three color channels (red, green, and blue).
- The U-Net starts with a series of convolutional layers, which are like filters that extract features from the image.
- These convolutional layers are followed by activation functions, which help to introduce non-linearity and make the network more powerful. 
- The U-Net also uses max pooling, which reduces the size of the image while preserving important features.
- After the first half of the network, the U-Net starts using transpose convolutional layers, which help to increase the size of the image again.
- The U-Net also includes skip connections, which copy the activations from earlier layers and combine them with the current layer's activations. This helps to preserve important information and improve the network's performance.
- Finally, the U-Net uses a one-by-one convolutional layer to map the output to a segmentation map, which assigns each pixel in the image to a specific class or category.

![u net](https://github.com/user-attachments/assets/f9c988ef-3164-4cf0-8a56-58931f97f062)