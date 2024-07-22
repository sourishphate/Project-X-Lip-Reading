# What are Neural Networks?
- Neural networks are a class of machine learning models inspired by the human brain. They consist of interconnected layers of nodes (neurons) that process data and learn to recognize patterns.
- Neural networks are computational models that mimic the way the human brain processes information.

## Neural Network representation:
- Input Layer: The first layer that receives the input data or parameters.
- Hidden Layers: Layers between the input and output layers where computations are performed and the model processes the input parameters. There can be multiple hidden layers.
- Output Layer: The final layer that produces the output.

![NN_REP](https://github.com/user-attachments/assets/7eb4fe21-7b56-4ab7-86bd-2a48c265f272)

## Computing Neural Network Outputs:
- Each node in the hidden layer of the neural network performs two computations:
  - The first one is that they compute 'z' for that layer using the formula involving weights and biases as shown below.
  - The second is by use this 'z' to compute the activation 'a' i.e the output for that layer using the  by passing 'z' to the sigmoid function .This acts as the input for similar computations in the subsequent layers.
- To optimize this i.e not using an explicit for loop to perform the computations we use vectorization for the input features , the weights and biases, the z and activations for each node the hidden layers.This makes the computations faster and easier.
- _strong When we have different nodes in a layer we stack them vertically.strong_

![Comp NN out (2)](https://github.com/user-attachments/assets/345c2ded-b9cb-4886-82ae-822b35b22355)

![Comp NN out (1)](https://github.com/user-attachments/assets/e3e384d1-c8c8-4207-8d0c-093c6d63cd1a)

## Vectorizing across multiple examples:

![Vectorizing across multiple examples](https://github.com/user-attachments/assets/8e2eb628-4d22-4058-a74c-74ffb3638f1a)

## Activation Functions:
An activation function is a mathematical function applied to the output of each neuron in a neural network. It determines whether a neuron should be activated or not by introducing non-linearity into the network, which allows it to learn and model complex data.

### Why are Activation Functions Needed?
- Non-linearity: Without activation functions, a neural network would simply perform linear transformations, limiting its ability to learn from complex patterns in data.

- Centering Data: The tanh (hyperbolic tangent) function often works better than the sigmoid function because it centers the data around zero, which can make learning easier for subsequent layers. This is similar to centering input data to have zero mean before training a model.

#### Why Your Neural Network Needs a Nonlinear Activation Function

- For a neural network to compute complex functions, nonlinear activation functions are necessary.
- Without a nonlinear activation function, a neural network is effectively performing a linear transformation of the input.
- A network with only linear activation functions is no more expressive than a single-layer model.
- Nonlinear activation functions enable the network to capture complex patterns by introducing nonlinearity at each layer.

#### Sigmoid vs. Tanh:

- The `tanh` function often works better than the sigmoid function for hidden units because it centers the data around zero, making learning easier for the next layer.
- The exception for using the sigmoid function is in the output layer for binary classification, where the output needs to be between 0 and 1.

_strong Gradient Issues: Both sigmoid and tanh functions can cause gradients to vanish when inputs are very large or very small, slowing down learning. ReLU and its variants are often preferred to avoid this problem.strong_
⁡
#### ReLU (Rectified Linear Unit):

- Introduced as a popular activation function.
- Defined as `ReLU(x) = max(0, x)`.
- The derivative is 1 for positive `x` and 0 for negative `x`.
- While technically the derivative at `x = 0` is not well-defined, in practice, it works fine by assuming either 1 or 0.

#### Leaky ReLU:

- An alternative to ReLU to avoid the dying ReLU problem, where the function allows a small gradient for negative inputs.
- Defined as `Leaky ReLU(x) = max(0.01x, x)`.
- The slope is a small constant (0.01) when `x` is negative.

#### Softmax:

- Used for multi-class classification problems.
- Converts logits into probabilities that sum to 1.
- Defined as `Softmax(x_i) = e^(x_i) / sum(e^(x_j))` for all `j`.
- Useful when the model needs to output a probability distribution over multiple classes.

```
### Choice of Activation Function:

- Sigmoid is recommended for the output layer in binary classification tasks.
- Softmax is recommended for the output layer in multi-class classification tasks.
- For hidden layers, ReLU is generally the default choice due to its efficiency and effectiveness.
- tanh can be used in hidden layers for zero-centered data.
- Leaky ReLU is useful to mitigate the dying ReLU problem.

```

####  Derivatives of Activation Function

- Sigmoid Function: `g(z)*(1 - g(z))`
- Hyperbolic Tangent (Tanh): `1 - g(z)^2`
- ReLU and Leaky ReLU: `0 or 1`

## Gradient Descent in Neural Networks

![gradient desc](https://github.com/user-attachments/assets/5b474814-bd19-4772-9710-e0335bd9c712)

1. **Initialization**:
   - Start with random values for weights and biases:
     - `W1`, `B1` for the hidden layer.
     - `W2`, `B2` for the output layer.

2. **Forward Propagation**:
   - Pass input data through the network to get predictions.
   - Calculate intermediate values and apply activation functions.

3. **Compute Gradients**:
   - Determine how much each weight and bias affects the prediction error.
   - This tells you how to adjust them to reduce error.

4. **Update Weights**:
   - Adjust `W1`, `B1`, `W2`, and `B2` using the computed gradients.
   - Use a learning rate to control the size of the adjustments.

### Forward Propagation

1. **Calculate Activations**:
   - **Hidden Layer**:
     - `Z1 = W1 * X + B1`
     - `A1 = Activation(Z1)` (e.g., ReLU, Sigmoid)
   - **Output Layer**:
     - `Z2 = W2 * A1 + B2`
     - `A2 = Sigmoid(Z2)` (For binary classification)

### Back Propagation

1. **Compute Gradients**:
   - **Output Layer**:
     - `DZ2 = A2 - Y` (Error in prediction)
     - `DW2 = DZ2 * A1^T` (Gradient for weights)
     - `DB2 = DZ2` (Gradient for biases)

   - **Hidden Layer**:
     - `DZ1 = (W2^T * DZ2) * Activation'(Z1)` (Error propagated back)
     - `DW1 = DZ1 * X^T` (Gradient for weights)
     - `DB1 = DZ1` (Gradient for biases)

### Training

- **Iterate**:
  - Repeat forward propagation and back propagation.
  - Update weights and biases each iteration to improve network performance.

## Random Initialization:
- One way would be to initialize the weights and biases to 0.While this won't cause any problems with the bias we will face difficulty with the weights.
- This is because if all weights are initialized to 0 then all the nodes in the hidden layers be computing the same function (i.e they will be symmetric) , no matter how long we train the neural network.
- There is not point in having multiple neurons if they perform the same function since we need them to compute different functions.




