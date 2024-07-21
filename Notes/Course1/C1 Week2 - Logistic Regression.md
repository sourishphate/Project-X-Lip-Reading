### How does computer represent an image?
- An image is stored using three matrices for the red, blue and green colour channels which together form the image .
- To store the image, all the pixel intensity values in the three colour channels are stored together in a feature vector variable .

## Binary classification 
- This classification gives us one of two possible outputs for given input parameters.This means that the output which is the dependent variable has two possible outcomes.
- Each instance in the dataset is assigned to one of these two classes based on a prediction model.
- The two possible outcomes are often labeled as 0 and 1, True and False, or positive and negative.

### Notation for training examples:
Examples are represented by (x,y)

Where x is the input feature vector of size nx 

nx =  No. of channels * size of the matrix for each channel

y is a binary output having values {0,1}.

m training examples of the form (x,y ) are put together in a set to form a training set.

In a similar manner m test examples can be put together to make test sets.

Now,

The feature vectors in each training example are put together in a matrix as columns .

So this Matrix will have the columns as the m input feature vectors from the training set and it will 
have nx number of rows.

` X.shape = (nx, m)` `Y.shape = (1,m)`

### Logistic Regression 
- It is a statistical method used for binary classification, which means it predicts one of two possible outcomes for a given input rather than predicting something continuous like size. It does this by modeling the probability that a given input belongs to a particular category.
- The input information is multiplied by some numbers called parameters and added to a bias term
- These parameters and bias term are adjusted during the learning process to make our predictions as accurate as possible.
- Unlike linear regression, which predicts a continuous output, logistic regression predicts probabilities of the outcome that are bounded between 0 and 1. This is achieved using the logistic function (also known as the sigmoid function).
  
![sigmoid](https://github.com/user-attachments/assets/f251c026-cc53-42a7-bdfa-a806aefe6d2a)

#### Parameters of Logistic Regression
- Weights (Coefficients): The weights or coefficients w1,w2,...,wnw_1, w_2, ..., w_nw1​,w2​,...,wn​ determine the influence of each feature on the prediction. These are learned from the data during the training process.
- Bias (Intercept): The bias term bbb is a constant that allows the decision boundary to be adjusted without depending on the input features.

#### Loss Function
- A function that measures the difference between the network's prediction and the actual target value. Common loss functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.
- The loss function we will use in logistic regression is called the negative log likelihood loss function.
- It is represented as `-y*log(y_hat) - (1-y)*log(1-y_hat)`, where y is the actual value of our model and y_hat is the predicted probability.

#### Logistic Regression Cost Function:
- The cost function in logistic regression is used to determine how much error is present or how close our output is to the predicted output.

- While training the model we will try to find parameters weight and bias such that the cost function is minimum.

## Gradient Descent:
Gradient Descent is an algorithm used to minimize a function by finding the direction of the steepest descent. In the context of neural networks, this function is usually the loss function, which measures how well the model's predictions match the actual data. 

### Computation Graph:
A computation graph gives us different steps which tell us how to solve or compute a function. 
It visually represents the operations and variables involved in computing a function.

### Derivatives and Their Importance:
In the context of machine learning, particularly neural networks, derivatives help us understand how changes in one variable affect another variable. This is crucial for optimizing functions, such as minimizing the error in a model.

### Logistic regression derivatives:
- Objective: Find the best parameter values (weights and bias) to minimize the negative log-likelihood loss function in logistic regression.
- Initial Guess: Start with random initial values for weights and bias.
- Calculate Predictions: Compute predictions using the logistic regression model with current parameter values.
- Compute Loss: Evaluate the negative log-likelihood loss function using the predictions.
- Derivatives Calculation:
  - Loss Derivative w.r.t Predictions: Calculate how the loss changes with respect to the predictions.
  - Loss Derivative w.r.t Parameters: Use the chain rule to calculate how the loss changes with respect to each parameter (weights and bias).
- Parameter Update:
  - Compute Step Size: Multiply each derivative by a small learning rate to determine the step size.
  - Adjust Parameters: Subtract the step size from the current parameter values to get updated weights and bias.
- Iterate: Repeat steps 3-6 until the parameter changes are very small or a maximum number of iterations is reached.
- Model Training: Through these iterations, gradient descent optimizes the parameters, reducing the loss and improving the model's prediction accuracy on new data.







