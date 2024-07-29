### 1. One Layered Convolutional network

- **Convolution:**

    - CNNs use a process called convolution, where they apply filters to an input image.
    - These filters help to extract important features from the image, such as edges or textures.

- **Non-linearity:**

    - After the convolution step, a non-linear function is applied to the filtered image.
    - This helps to introduce non-linear relationships between the features, allowing the network to learn more complex patterns.

- **Pooling:**

    - CNNs also use a technique called pooling, where they reduce the size of the image by selecting the most important features.
    - This helps to make the network more efficient and reduces the risk of overfitting.

- **Fully Connected Layers:**

    - Finally, the features extracted from the previous steps are passed through fully connected layers, which perform classification or regression tasks based on the extracted features.

![one layered convo](https://github.com/user-attachments/assets/3d40390b-2515-4a9e-93ec-2e6875ea2b37)

#### 1.1 Notation

- `f[l] = filter size`
- `p[l] = padding`
- `s[l] = stride`
- `nc[l] = number of filters`
- `Each filter: f[l] * f[l] * nc[l-1]`
- `Activation: a[l] = nh[l] * nw[l] * nc[l]`
- `Weights: f[l] * f[l] * nc[l-1] * nc[l]`
- `Bias: nc[l] - (1,1,1,nc[l])`
- `n[l] = ((n[l-1] + 2p[l] - f[l]) / s[l] ) + 1`
- `Input = nh[l-1] * nw[l-1] * nc[l-1]`
- `Output = nh[l] * nw[l] * nc[l]`

### 2. Simple Convolution Network Example (ConvNet)

![convnet](https://github.com/user-attachments/assets/4b86c227-88b8-4cdf-a74a-54fd65d879a8)

#### 2.1 Types of Layer

- Convolution (CONV)
- Pooling (POOL)
- Fully Connected (FC)

#### 2.2 Pooling Layer

- Max pooling

    - In max pooling, the maximum value within each region is selected as the representative value.
    - It focuses on capturing the most prominent or significant feature within the region.
    - Max pooling helps in preserving the strongest features detected in the input data.

![Max pooling ](https://github.com/user-attachments/assets/24cfbe09-495e-43f3-b8ee-0480f3f88360)

- Average Pooling

    - In average pooling, the average value of all the numbers within each region is selected as the representative value.
    - It takes into account the overall intensity or magnitude of the features within the region.
    - Average pooling helps in reducing the impact of outliers or extreme values in the input data.
    - It is sometimes used in very deep neural networks to collapse the representation of a large volume into a smaller one.

\*Hyperparameters: `filter size` `stride`

![average pooling](https://github.com/user-attachments/assets/08ff01f3-fc72-463f-931e-24628691fb28)

### 3. Convolutional Neural Network Example (LeNet-5)

1. **Input:**

- We start with an image that is 32 x 32 pixels and has 3 color channels (RGB).

2. **Convolutional Layer:**

- We apply a filter of size `5 x 5` to the input image. This filter helps us extract features from the image.
- We use 6 filters in this layer, so the output becomes `28 x 28 x 6`

3. **Pooling Layer:**

- We apply a pooling operation to reduce the size of the output from the previous layer.
- In this case, we use max pooling with a filter size of `2 x 2` and a stride of 2.
- This reduces the output size to `14 x 14 x 6`

4. **Repeat:**

- We can repeat the convolutional and pooling layers to further extract features and reduce the size of the output.

5. **Fully Connected Layer:**

- Finally, we connect the output of the previous layers to a fully connected layer, which helps us classify the image into one of the 10 possible digits (0 to 9).

![neural network example](https://github.com/user-attachments/assets/b1b15f8f-8688-41c6-82e1-a8954ba7f26b)

![nn example 2](https://github.com/user-attachments/assets/dcd61d0d-76a4-4653-bbed-a04ead65655a)

#### 3.1 Why convolution?

- **Parameter sharing:**

Same set of parameters can be used in different parts of an image to detect certain features.

- **Sparsity of connections:**

Each output in a CNN only depends on a small number of inputs. This allows the network to focus on important features and ignore irrelevant ones.

![why Convolutions](https://github.com/user-attachments/assets/0531cbce-9575-45a5-93b4-592ca5117102)
