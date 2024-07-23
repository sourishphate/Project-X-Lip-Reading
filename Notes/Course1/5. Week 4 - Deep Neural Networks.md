# Deep Neural Networks:
- A deep neural network is a machine learning model with multiple layers between the input and output layers. These intermediate layers are called hidden layers.
- The presence of multiple hidden layers makes the model more complex and is able to make better predictions

## Matrix Dimensions:
- The dimension of a matrix for parameters(weights) for a layer depend on the the number of neurons in that layer as well as the layer before it.
- For example :The weight matrix for the first hidden layer (Layer 1) will be 3x2, because it needs to transform a 2-dimensional input into a 3-dimensional output.

![Matrix dim](https://github.com/user-attachments/assets/40f6a01e-41be-4269-9547-d95dd3f8baaa)

## Building Block of Deep Neural Netrwork:
- Building a neural network involves two main steps i.e forward propagation and back propagation.
- Forward propagation:
  - Forward propagation is the process of moving from the input layer through the hidden layers to the output layer, calculating the output at each layer and using it as input in the next layer.

![Forward propagation](https://github.com/user-attachments/assets/ca121efd-c577-4791-b5b6-998b44597bc0)

- Backward Propagation:
  - The process of updating the weights and biases to minimize the loss function.
  - It strats by computing the difference betwee the predicted output and the actual output. 
  - It involves calculating the gradient of the loss function with respect to each weight and bias, and then adjusting them in the opposite direction of the gradient (gradient descent).
 
![back propagation](https://github.com/user-attachments/assets/9ba45e11-7512-4eea-93f5-24fa3fbc7baa)

### Step-by-Step Process:
- Forward Propagation:
   - You start by calculating the output of the neural network given the current parameters. 
   - This involves passing the input data through each layer of the network to get the final output (prediction).
   - The output of each layer is used as the input for computations in the next layer.

- Calculate Loss:
  - Compare the network’s prediction with the actual target values to compute the loss. 
  - The loss is a measure of how far off the predictions are from the actual values.

- Backward Propagation:This is where gradient calculation happens. You calculate how the loss changes with respect to each parameter in the network. This involves:
  - Partial Derivatives: Finding the partial derivatives of the loss with respect to each parameter (weights and biases). These partial derivatives are the gradients.
  - Chain Rule: Applying the chain rule from calculus to propagate the gradients backward through the network, from the output layer to the input layer.

 ![Bulding nn](https://github.com/user-attachments/assets/9c5a13b6-fee2-4df3-8990-0dcd85735e07) 

### Parameters :

- Parameters are functions that are used to manipulate the input data while it passes throug the different layers in the neural network. This includes weights and biases.

### Hyperparameters:
- Hyperparameters are settings that control the training process and the structure of the neural network.They are not learned directly from the training data but are predefined and adjusted to optimize the model. 
- They control or determine the values of parameters 'w' and 'b'.
- Some key hyperparameters include `Learning rate `, `number of iterations` , `number of hidden layers` ,`number of hidden units`, `choice of  activation functions` etc.
- Other include Momentum Term, Mini-Batch Size , Regularization Parameters.
-Applying deep learning involves a lot of trial and error.Since the optimal values of hyperparameters can vary widely depending on the problem and the dataset, one typically has to experiment with different settings.