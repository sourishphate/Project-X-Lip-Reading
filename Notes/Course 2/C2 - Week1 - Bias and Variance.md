## Train/Dev/Test Sets:
- Datasets are split into trainig sets, development sets and test sets for efficient working.
- The dataset is split into 98% training, 1% development , and 1% test set when we have large datasets.If we have a small dataset the general 60/20/20 split also works as well.
- The general workflow is that we train the algorithms on the training set and the use the dev set to determine which model works best and after doing this long enough we use the test set to determine how well the model is doing.
- It is important to make sure that the training set and the dev/test sets are all from the same distribution. For example - If the model is trained on high resolution images but the test set contains low reswolution images the model will not work properly.
- It is ok in some cases to not have a teswt set and only a dev set.

## Bias And Variance:
- Bias is the error that is introduced when it is assumed that the model is too rigid or underfitting i.e the model is too simple.
- Variance is the error that is introduced when it is assumed that the model is too flexible overfitting i.e the model is too complex.It captures the noise as well.

- Training Set Error: Indicates how well the model fits the training data. High training error suggests high bias.
- Development (Dev) Set Error: Indicates how well the model generalizes to unseen data. A large gap between training and dev set error suggests high variance.

| Training Error | Dev Set Error  | Diagnosis                       |
|----------------|----------------|----------------------------------|
| High           | High           | High bias (underfitting)         |
| Low            | High           | High variance (overfitting)      |
| High           | Higher         | High bias and high variance      |
| Low            | Low            | Low bias and low variance (ideal scenario) |

#### Example:
|                           | Train set error | Dev set error |
| ------------------------- | --------------- | ------------- |
| High variance             | 1%              | 11%           |
| High bias                 | 15%             | 16%           |
| High bias & high variance | 15%             | 30%           |
| Low bias & Low variance   | 0.5%            | 1%            |

##### _strongBias-Variance Tradeoff: Balancing between a simple model (high bias, low variance) and a complex model (low bias, high variance) to achieve the best performance on new, unseen data.strong_

```
 - High Bias:
  - Make the network bigger i.e add more layers to the network.
  - Train the network for longer.

 - High Variance:
  - Get more data if possible to train the network.
  - Use regularization techniques.
```

