# Vectorization:
- The main idea behind vectorization is to avoid using explicit for loops when working with data for the neural networks to make implementation faster.
- This allows us work with entire arrays or matrices without having to iterate over each element individually which allows us to perform diferent computations much faster.
- Vectorization is quite helpful when working with large datasets.
- In a vectorized implementation, we directly compute the dot product without using an explicit for loop. This is done using built-in functions like `np.dot()` in Python or numpy.

![vectorization](https://github.com/user-attachments/assets/bd2ea637-80e8-4769-9c16-3ef767ce5b4a)

![vec eg](https://github.com/user-attachments/assets/22fd3ab0-5109-48dd-af83-e68584998ae6)


# Vectorizing Logistic Regression:
- Vectorizing logistic regression state of computing the predictions for each training example separately all inputs are stacked horizontally in a matrix 'X'. 
- Using  to which we can use vectorization to compute the predictions (z) for all values at once and then stack/store them into another matrix 'Z' , using a single line of code.
-Similarly all activations can be computed and stored without having to use an explicit for loop.

![VLR](https://github.com/user-attachments/assets/a1103c6f-cc7a-41cc-ade4-c5ae3b344a3b)

![VLR2](https://github.com/user-attachments/assets/3e716c02-8dab-4258-a1c7-2a114394e42b)

## Implementation of Logistic Regression Vectorization:

![Implement LR](https://github.com/user-attachments/assets/f22e8d03-bf86-4320-8cec-8203a460822d)

## Vectorizing across multiple examples:

![Vectorizing across multiple examples](https://github.com/user-attachments/assets/d21f4e3d-69af-4bb8-80c1-69183c3690f2)

### General Principle of broadcasting:

![Broadcasting](https://github.com/user-attachments/assets/8c54cbf1-b0bf-4c8a-83ee-c9b90467e5b8)








