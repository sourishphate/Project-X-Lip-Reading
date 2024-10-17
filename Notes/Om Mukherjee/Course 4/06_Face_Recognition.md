### 1. Face Verification v/s Face Recognition

| Verification | Recognition|
| --- | --- |
| Input image, name/id |  Has a database of K persons|
| Output whether the input image is that of the claimed person|  Involves identifying a person's face among a large database of faces. |
| When the system checks if a given face matches the claimed identity.| Output id if the image is any of the k person |

### 2. One Shot Learning

The challenge is to be able to recognize a person with just one single image or example of their face. This is because we often have only one picture of each person in our database.

Instead of trying to train a neural network with a small training set, we use a different approach called "learning a similarity function.

- This function takes two images as input and outputs a number that represents the degree of difference between the two images.
- During recognition time, we compare a new picture with the images in our database using this similarity function.
- If the degree of difference `d(img1,img2)` is less than a certain threshold, we predict that the two pictures are of the same person.
- If the degree of difference `d(img1,img2)` is greater than the threshold, we predict that they are different persons.

![one shot learning](https://github.com/user-attachments/assets/5e91d351-a693-4578-9283-9f14a602039b)

### 3. Siamese network

The goal is to input two images of faces and determine how similar or different they are. To do this, we use a neural network that takes an image as input and produces a vector of numbers that represents the image. This vector is called an `encoding`

- To compare two images, we feed them into the same neural network and get two different encodings.
- Then, we calculate the difference between these encodings. 
    - If the images are of the same person, we want the difference to be small.
    - If the images are of different people, we want the distance to be large.

![siamese network](https://github.com/user-attachments/assets/1620ca89-9515-4519-b4a0-981546fcba4e)

### 4. Triplet Loss

Triplet loss is a loss function used in deep learning models, specifically for tasks like face recognition. The goal of triplet loss is to learn a good encoding for images, so that similar images have similar encodings and different images have different encodings.

- To apply triplet loss, we compare triplets of images: an anchor image, a positive image (which is similar to the anchor), and a negative image (which is different from the anchor).
- The objective is to minimize the distance between the anchor and positive encodings, while maximizing the distance between the anchor and negative encodings.

`L(A, P, N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + α, 0)`

*Here, ||x||^2 represents the squared Euclidean distance of vector x, and α is a margin parameter that determines the desired separation between the positive and negative pairs.*

![Loss ](https://github.com/user-attachments/assets/f3912d21-3aea-4cd9-93d8-fdfe54a54799)

#### 4.1 Choosing the triplets A, P, N

- **Random selection:** 
    - One approach is to randomly select triplets from your training set. However, this can lead to many triplets being too easy for the model to learn from, resulting in limited improvement.
    - Random selection may not provide enough challenging examples for the model to optimize effectively.

- **Semi-hard selection:**
    - A more effective approach is to choose triplets where the distance between the anchor (A) and positive (P) encodings is close to the distance between the anchor (A) and negative (N) encodings, but still greater than a certain margin value (α). 
    - This forces the model to work harder to optimize the encodings and improve the separation between similar and dissimilar images.

- **Online selection:** 
    - Another strategy is to dynamically select triplets during the training process. This involves selecting triplets based on the current state of the model.
    - For example, you can choose triplets that the model is currently struggling with or misclassifying. This adaptive selection helps the model focus on challenging examples and improve its performance over time.

![choosing triplet](https://github.com/user-attachments/assets/42ad6aba-2d6c-4405-ab6e-1f1e7ceef469)

### 5. Face Recognition as a Binary Classification

![binary classifi](https://github.com/user-attachments/assets/071f3f52-45e8-452c-a467-a19cbc6576e0)
