### 1. LeNet-5

- It takes a grayscale image of size 32 by 32 pixels as input.
- The network applies a set of filters to the image, which helps to extract features.
- Then, it uses pooling to reduce the dimensions of the image.
- After that, another set of filters is applied, followed by more pooling.
- Finally, there are fully connected layers that connect all the nodes together, leading to the final output.

*It has 64k parameters*

![LeNet](https://github.com/user-attachments/assets/b112c2a7-d0d8-42ae-867a-eb3fd1c5becd)

### 2. AlexNet

- It takes color images of size 227 by 227 pixels as input.
- It also applies filters and pooling layers, but with different sizes and strides.
- It has more layers and parameters, which allows it to achieve better performance in image classification tasks.

*It has 60M parameters, multiple ReLU, multiple GPUs*

![alexnet](https://github.com/user-attachments/assets/359ac5b7-cc3b-4640-b24d-8679dcb97868)

### 3. VGGNet

- It only uses 3x3 filters with stride of 1 and 2x2 pooling layers with stride of 2.
- It has a uniform structure with multiple convolutional and pooling layers.

*It has 138M parameters*

![vgg net](https://github.com/user-attachments/assets/37a70874-f4cf-40e8-87e4-f613af54f5e8)

### 4. ResNet

#### 4.1 Skip connections

When we have a deep neural network with many layers, it can be difficult to train because of vanishing and exploding gradient problems. 

- Skip connections allow us to take the activation from one layer and feed it to another layer that is much deeper in the network.
- With skip connections, we can take the activation from the first layer and directly feed it to the final layer, bypassing the intermediate layers. `a[l+2] = g(z[l+2] + a[l])`
- By using skip connections, we can build a type of neural network called a ResNet.

#### 4.2 Residual network

![resnet](https://github.com/user-attachments/assets/5e756653-07f5-4800-803e-69c5d69b58c6)

#### 4.2 Why does resnet works?

- Instead of just adding new layers, if we include a shortcut connection called a residual block allows the network to copy the activation from a previous layer and add it to the output of the new layers.
- These extra layers can easily learn the identity function, which means they can simply copy the activation from the previous layer and pass it on to the next layer.
- This makes it easier for the network to learn and doesn't hurt its performance. In fact, sometimes it even improves the network's performance because these extra layers can learn something useful in addition to the identity function.

*The key idea behind ResNets is that they make it easy for the network to learn the identity function, which is why adding extra layers doesn't harm the network's performance.*

*You can also add `Ws` to make input dimension of two connection equal* 

![why resnet work](https://github.com/user-attachments/assets/43d31cf7-810e-49c0-8d48-223f522389d5)

![Resnet2](https://github.com/user-attachments/assets/c0a1a7bb-7474-4333-a93b-d72636cfb736)

### 5. Network in Network (1*1 Convolution)

If you have an image with multiple channels, a one-by-one convolution can do something more interesting.
- It takes each position in the image, multiplies the corresponding values in each channel by a set of weights, and then applies a nonlinearity function (ReLU) to the result. 
- This allows you to perform more complex computations on the image.

*It is capable of shrinking channels* 

![convolution1](https://github.com/user-attachments/assets/e5bdf03e-8244-4189-a8bc-36dc62e91a7a)

### 6. Inception Network

Instead of choosing just one filter size or pooling layer, the Inception module uses multiple filter sizes and pooling layers. It applies different filters to the input volume and stacks up the outputs. 

- The Inception module can take an input volume and output a volume with different dimensions.
- It concatenates the outputs of the different filters and pooling layers.This makes the network architecture more complex, but it has been shown to work really well in practice.
- The Inception module reduces the need to commit to a specific filter size or pooling layer and lets the network learn the best combinations.

![inception ](https://github.com/user-attachments/assets/6e29f800-f6d3-4670-a195-25b6371edb78)

#### 6.1 Computation cost problems

- When using a 5x5 filter in the Inception module, the number of multiplications and additions needed to compute the output can be quite high. 
- This can lead to longer training times and increased resource requirements, making the network slower and more computationally expensive. `120M parameters`

![comp cost prob](https://github.com/user-attachments/assets/01714425-a2d6-4a2b-8cc2-e1ca1241a41b)

#### 6.2 Using 1*1 convolution

To address this problem, the Inception module introduces the concept of bottleneck layers and 1x1 convolutions.

- By using a 1x1 convolution to reduce the number of channels before applying larger filters, the computation cost can be significantly reduced without sacrificing performance.
- This allows for more efficient training and inference in the network.`12.4M parameters`

#### 6.3 Case Studies

- **Inception Module:** 
    - It takes the output from a previous layer as input and applies different types of convolutions to it. 
    - These convolutions can have different filter sizes and depths, allowing the network to capture different levels of detail in the image.

- **Concatenation:** 
    - After applying the convolutions, the outputs are concatenated together to create a single output. 
    - This allows the network to capture a wide range of features at different scales and resolutions.

- **Repeated Blocks:**
    - The Inception network consists of multiple repeated blocks, each containing several inception modules.
    - These blocks are stacked together to create a deep network that can learn complex patterns in images.

- **Side-Branches:**
    -  In addition to the main network, the Inception architecture also includes side-branches. 
    - These side-branches take hidden layers from the network and use them to make predictions. This helps regularize the network and prevent overfitting.

By combining these building blocks, the Inception network can learn to recognize a wide variety of objects in images.

![inception ase study](https://github.com/user-attachments/assets/9785525a-05b5-4ae7-bf2d-ac1124829699)

![inception case study 2](https://github.com/user-attachments/assets/ae62e10c-7e28-4d12-979f-5fd96e08df0d)

### 7. MobileNet

MobileNets are designed to be used in low-compute environments, such as mobile phones, where there is limited processing power. The main idea behind MobileNets is to reduce the computational cost of the network while still maintaining good performance.

#### 7.1 Normal v/s depthwise seperable convolution

- **Normal convolution**
    - In a normal convolution, you have an input image and a filter. The filter is moved across the image, performing multiplications and additions to produce an output. This process can be computationally expensive.

![mobile net normal conv](https://github.com/user-attachments/assets/161db848-c617-42a1-93e6-f2c4113b2dfe)

- **Depthwise separable convolution**
    - Instead of using one filter for all the channels in the input image, MobileNets use separate filters for each channel. 
    - This reduces the number of multiplications needed. Then, a pointwise convolution is applied to combine the outputs of the separate filters.

![mobile net normal conv](https://github.com/user-attachments/assets/161db848-c617-42a1-93e6-f2c4113b2dfe)

![poitnwise conv](https://github.com/user-attachments/assets/cfd9a557-890c-4732-9ae8-5608437de52f)

*By using depthwise separable convolution, MobileNets can achieve similar results to normal convolutions but with fewer computations.*

![depthwise seperable conv](https://github.com/user-attachments/assets/8e0e86f0-57be-46e6-9f28-942ce7dfc420)

#### 7.2 Cost Summary

![mobile net summary cost](https://github.com/user-attachments/assets/26e48146-4229-4e51-98c6-873e86c9a159)

*MobileNet is 10 times more efficient in computation*

#### 7.3 Architecture

In MobileNet v2, there are two main changes. 
- It adds a residual connection, which allows information to propagate backward more efficiently.
- It introduces an expansion layer before the depthwise convolution, followed by the pointwise convolution.

![mobilenet architecture](https://github.com/user-attachments/assets/75e5ca57-204e-4416-9511-cb94c3ecd774)

The MobileNet v2 architecture consists of repeating a block `17 times` called the bottleneck block multiple times. Each bottleneck block includes the expansion layer, depthwise convolution, and pointwise convolution. 

![mobile net bottleneck](https://github.com/user-attachments/assets/0a5efe93-87d8-437d-8528-2afedb0cbe18)

### 8. EfficientNet

![efficient net](https://github.com/user-attachments/assets/09d1c1da-fcf0-428c-b5f5-6e927aee6545)

### 9. Practical advice on using ConvNet

#### 9.1 Transfer learning

Transfer learning is a technique used in computer vision to make faster progress when building a new computer vision application. Instead of starting from scratch and training a neural network from random initialization, transfer learning allows us to use pre-trained networks that have already been trained on large datasets by other researchers.

- Transfer learning is a powerful technique because it allows us to leverage the knowledge and insights gained from training on large datasets.
- It saves us time and computational resources by using pre-trained networks as a starting point for our own specific tasks. 
- Whether we have a small or large training set, transfer learning can help us achieve better performance in computer vision applications.

![transfer learning](https://github.com/user-attachments/assets/1e6c7a73-05b0-4b64-bb50-e6633c8d02e9)

#### 9.2 Data augmentation

In computer vision, data augmentation is a technique used to improve the performance of computer vision systems by increasing the amount of training data. Computer vision is the task of analyzing and understanding images, which can be quite complex. To do this, we need to learn a complicated function that can recognize objects or patterns in images.

Data augmentation involves creating new training examples by applying various transformations to the existing images in our dataset.

- **Geometric transformations**
    - It include techniques like `mirroring` `random cropping` `rotation` and shearing.
    - These transformations change the `position` `size` or `shape` of the image.

![geometric transformation](https://github.com/user-attachments/assets/13282bfb-c7a2-445d-ac64-509b10899a3c)

- **Color shifting**
    - It involves changing the colors of the image by adding or subtracting values from the `red` `green` and `blue` channels.

![color shifting](https://github.com/user-attachments/assets/0ae9b17b-6737-4770-a1dd-fc721fbd063a)

By applying these transformations, we can create new variations of the original images, which helps our models learn to recognize objects or patterns from different perspectives. 

*This makes our models more robust and better able to handle variations in real-world images.*



