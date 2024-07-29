### Edge Detection

- To detect edges, we use a technique called convolution. It involves taking a small matrix called a filter and sliding it over the image.
-  The filter has numbers that represent the weights of each pixel. By multiplying the filter values with the corresponding pixels in the image and adding them up, we get a new matrix that represents the detected edges.

![Vertical edge detection](https://github.com/user-attachments/assets/caf8ade6-2719-4582-9fd4-ec769ef4ea0e)

#### Vertical edge detection

- If we use a filter that detects vertical edges, it will have positive values on the left side, zero values in the middle, and negative values on the right side.
- When we convolve this filter with the image, we get a new matrix that highlights the vertical edges in the image.

![Vertical edge detection 2](https://github.com/user-attachments/assets/f234360d-90f4-46ee-95f2-83df8bf643ab)

#### Positive and negative edge detection

- Positive edge:
    - A positive edge represents a transition from a darker region to a brighter region.
    - It occurs when the intensity values increase as you move across the edge.
- Negative edge:
    - A negative edge represents a transition from a brighter region to a darker region.
    - It occurs when the intensity values decrease as you move across the edge.

*By understanding the difference between positive and negative edges, we can analyze images and identify the direction of intensity changes*

#### Learning to detect edges

![Learning to detect edges](https://github.com/user-attachments/assets/44f94812-165f-490b-b83b-38b3e2da47d3)

### Padding

#### Problem 

1. When we apply a convolutional operator, the image size shrinks. This can be a problem if we have many layers in our neural network because the image can become very small. 
2. We convolve an image with a filter, the pixels at the corners and edges of the image are used less in the output. This means we're throwing away important information from the edges of the image.

*To solve these problems, we can pad the image.*

#### Solution

- Padding means adding an extra border of pixels around the edges of the image.
- By doing this, we can preserve the original input size of the image and ensure that the pixels at the corners and edges are used more effectively.

There are two common choices for padding: `valid convolution` and `same convolution`.\
*Valid convolution means no padding, while same convolution means padding the image so that the output size is the same as the input size*

![Padding](https://github.com/user-attachments/assets/c749d13c-4999-4df3-bf99-8b386d5dc423)

### Strided Convolutions

Normally, you would slide the filter over the image one step at a time and perform calculations. But with stride convolutions, instead of moving the filter one step at a time, you move it two steps at a time.

1. **Dimensionality reduction:**

- Stride convolutions can reduce the spatial dimensions of the input feature maps.
- By moving the filter with larger strides, the output feature map size is reduced, which can help in reducing the computational complexity of the network.

2. **Faster computational:**

- With larger strides, fewer computations are required compared to traditional convolutions. 
- This can lead to faster training and inference times, especially when dealing with large datasets or complex models.

3. **Increased receptive field:**

- Stride convolutions allow the network to capture information from a larger area of the input image.
- By skipping some positions during the convolution operation can be beneficial for tasks that require capturing global context or detecting larger patterns.

4. **Feature extraction at different scales:**

- By using different stride values in different layers of the network, it is possible to extract features at multiple scales.
- This can be useful for tasks such as object detection, where objects of different sizes need to be detected.

5. **Regularization:**

- Stride convolutions can act as a form of regularization by reducing the spatial resolution of the feature maps.
- This can help prevent overfitting and improve the generalization ability of the network.

![strided convolution](https://github.com/user-attachments/assets/d50103ec-a08b-4c50-9ef3-fc1a11108504)

### Convolutions Over Volume

- To detect features in this image, like edges or other patterns, we can use a 3D filter.
- This filter also has three layers corresponding to the red, green, and blue channels. We place this filter over the image and perform a convolution operation
- This means we multiply each number in the filter with the corresponding numbers in the image, add them up, and get an output.
- We slide the filter over the image and repeat this process to get the complete output.

![convo over vol](https://github.com/user-attachments/assets/94423389-e93d-4ad9-bcad-8ebd9b2b04a4)

#### Multiple filters

![multiple filter](https://github.com/user-attachments/assets/1048a557-fbd8-433f-822e-a49da4412bc5)
