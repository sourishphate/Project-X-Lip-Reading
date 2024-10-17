### 1. Neural Style Transfer

Neural Style Transfer is a fun and exciting application of Convolutional Neural Networks (ConvNets). It allows you to create new artwork by combining the content of one image with the style of another image.

![neural style transfer](https://github.com/user-attachments/assets/26f349a5-eb7b-498f-9044-581ff407ea1c)

### 2. Deep ConvNet Learning

ConvNets are a type of neural network that are commonly used for image recognition tasks. The deeper layers of a ConvNet are responsible for detecting more complex features and patterns in images.

- As we move to deeper layers, the hidden units start detecting more complex shapes, textures, and even objects.
- For example, in layer 2, hidden units might be looking for vertical textures or round shapes. 
- In layer 3, hidden units might be detecting cars, dogs, or even people.
- And as we go even deeper, the hidden units can detect more specific objects like different breeds of dogs or even keyboards and flowers.

*They start with simple features like edges and colors and gradually learn to detect more intricate patterns.*

![deep convnet learning](https://github.com/user-attachments/assets/28d95a33-307f-47eb-9ccc-098f241229b5)

![visualizing deep layer 2](https://github.com/user-attachments/assets/b64b0abd-9767-44f3-84fb-c82ff856fd80)

### 3. Cost Function

Cost function measures how good a generated image is. We use gradient descent to minimize this cost function and generate the desired image.

**Content Cost:** It measures how similar the content of the generated image is to the content of the original image.

`Jcontent(C,G) = 1/2 * ∑(a[l][C] - a[l][G])^2`

![content cost](https://github.com/user-attachments/assets/460a4dff-47ba-4922-9050-20ad7398df7e)


**Style Cost:** It measures how similar the artistic style of the generated image is to the style of the style image.

`Jstyle(C,G) = ∑∑(S^L(k, k') - G^L(k, k'))^2`

Cost function: `J(G) = α J_content(C, G) + β J_style(S, G)`

![style cost](https://github.com/user-attachments/assets/88121e5d-8c8c-41b1-a9dd-fdd14850eb28)


![cost function](https://github.com/user-attachments/assets/78f7b986-1a0c-44bd-ab3e-9b8bdaaa44e2)

- To generate the image, we start with a randomly initialized image and update its pixel values using gradient descent
- As we minimize the cost function, the image gradually becomes more similar to the content image in terms of its content and more similar to the style image in terms of its style.

*By adjusting the weights of the content cost and style cost, we can control the balance between content and style in the generated image.*

![cost function2](https://github.com/user-attachments/assets/305f0160-b23e-4db4-bab0-0f90f5a2e1d1)

#### 3.1 Style matrix

- Style matrix is computed by measuring the correlation between activations across different channels in layer L. It is an nc by nc dimensional matrix, where nc is the number of channels in that layer.
- Each element in the style matrix, denoted as `G_l(k, k)`, measures how correlated the activations in channel k are with the activations in channel k'. The values of k and k' range from 1 to nc.
- To compute the style matrix, we sum the products of activations at different positions (i, j) in the block for channels k and k'. This is done for all positions and all pairs of channels.
- The resulting style matrix captures the correlations between different channels in layer L, providing a measure of the style of the image.

`G^L(k, k') = ∑∑ activation(i, j, k) * activation(i, j, k')`

![style matrix](https://github.com/user-attachments/assets/f6a107cf-386d-41ba-a28a-938e9deb3edf)