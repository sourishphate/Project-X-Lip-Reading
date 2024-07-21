# Neural Network

- Input goes through a activation function (neuron) to produce desired results
- In simpler terms, the network of this neuron is known as neural network

#### Supervised Learning

- In supervised learning, we have some input data and we want to learn a function that can map that input data to an output.

_Standard Neural Network_: Real Estate, Online Advertising \
_Convolutional Neural Network_: Photo Tagging \
_Recurrent Neural Network_: Speech Recognition, Machine translation

    Why is Deep Learning taking off?
    1. Availability of large amounts of data.
    2. Ability to train large neural networks.
    3. Algorithmic innovations

### Binary Classification

- We represent the image as a feature vector
- We then use logistic regression to learn a classifier that can take this feature vector as input
- Predict whether the image is a cat or not.

_We'll use lowercase m to represent the number of training examples, and lowercase n to represent the dimension of the input feature vector.
` X.shape = (nx, m)` `Y.shape = (1,m)`_

## Logistic Regression

![Logistic regression](https://github.com/user-attachments/assets/f2438f7f-db07-43c7-b008-405e082e0087)

- The input information is multiplied by some numbers called parameters and added to a bias term
- These parameters and bias term are adjusted during the learning process to make our predictions as accurate as possible.
- Logistic regression helps us estimate the chance of an event happening based on certain input features.
- It uses a special function called the sigmoid function to convert the input into a probability value between zero and one

### Lost function

- To measure how well your model is doing, you need a loss function.
- The cost function tells you how close your predictions are to the actual labels in your training data.

![loss function](https://github.com/user-attachments/assets/097f8170-067a-47ba-80ca-74ceb9f37617)

_Here, y represents the true label (whether someone bought the product or not), and y hat represents the predicted output of your model._

#### Cost function

![Cost funtion](https://github.com/user-attachments/assets/3989260e-d68d-4bf7-b8d3-67597242413b)

- The cost function we use in logistic regression is called the negative log likelihood loss function
- It is defined as `-y*log(y_hat) - (1-y)*log(1-y_hat)`, where y is the true label and y_hat is the predicted probability.

### Gradient descent

![Gradient descent](https://github.com/user-attachments/assets/3085afa6-623d-4be6-b47d-80be0fbbd4ea)

- We need to find w, b that minimize cost function.
- Derivatives help us improve our neural networks. They allow us to adjust the parameters of the network, like the weights and biases

### Computation graph

![Computation Graph](https://github.com/user-attachments/assets/c1d5556a-0b40-430c-9b15-c99076a66dc2)

- It is a flowchart that shows the steps needed to compute a function.
- Each step in the graph represents a calculation or operation.

### Computing derivatives

![Computing derivative](https://github.com/user-attachments/assets/e42ef555-309d-4dc7-9182-208b858d8e3c)

- To calculate derivatives, we want to know how a small change in one variable affects the value of another variable.
- We can also use the chain rule in calculus to calculate derivatives.

### Logistic regression derivatives

![Logistic regression derivatives](https://github.com/user-attachments/assets/f8cb4022-8edf-42dd-afeb-e780d33c19e2)

- The main idea is to find the best values for the parameters (weights and bias) in logistic regression that minimize the loss function.
- To compute the derivatives, we go backward through the graph. We first calculate the derivative of the loss with respect to the predictions.
- Then, we use the chain rule to calculate the derivative of the loss with respect to the parameters.
- These derivatives tell us how much we need to change the parameters to reduce the loss.
- By using gradient descent, we can train the logistic regression model to make accurate predictions on new data

### Logistic regression on m examples

![logistic regression on m example](https://github.com/user-attachments/assets/f6a7cd4f-f1f0-4d3c-b8a9-b49a595d1c80)

- We initialize some variables to zero, then iterate over each training example.
- For each example, we compute the prediction, update the cost function, and calculate the derivatives.
- Finally, we divide the derivatives by the number of training examples to get the average
- We then update the parameters using the derivatives and repeat this process for multiple iterations to improve the performance of our algorithm.
