### 1. Object Localization

Object detection is a field in computer vision that focuses on identifying and locating objects within images or videos. It goes beyond just recognizing what an object is, but also involves drawing a bounding box around the object to indicate its position.

#### 1.1 Classification with localization

- Training a neural network to output not only the object class label but also the bounding box coordinates (`bx`, `by`, `bh`, `bw`), you can teach the algorithm to recognize and locate objects accurately.
- Object detection is a powerful technique that has numerous applications in various fields. 
-  It allows computers to understand and interact with the visual world, enabling tasks like autonomous driving, surveillance, and object recognition in images and videos.

![localization](https://github.com/user-attachments/assets/4fff74e9-38b4-4045-bb1d-85e5a6371455)

#### 1.2 Defining the target label y

![defining the target label y](https://github.com/user-attachments/assets/3dc2da5a-8636-4d9a-b713-ce876a350577)

### 2. Landmark Detection

Landmark detection involves identifying and locating specific points or landmarks in an image. These landmarks can be important features like the corners of someone's eyes or the edges of their mouth.

- We use a neural network that is trained to output the coordinates of these landmarks.
- By training the neural network with labeled data that contains the coordinates of these landmarks, we can teach it to recognize and locate them accurately.

*Landmark detection is a fundamental building block for various applications like face recognition, emotion recognition, and computer graphics effects.*

![Landmark detection](https://github.com/user-attachments/assets/f28b1127-270f-423b-968c-c3d279b83b09)

### 3. Object Detection

- Create a training set of images that have cars and images that don't have cars. These images should be cropped to only show the car, so there's no background clutter.
- Train a Convolutional Neural Network (ConvNet) using these cropped images. ConvNet's job is to take an image as input and output whether there's a car in it or not. 
- Once the ConvNet is trained, you can use it in the Sliding Windows Detection Algorithm.

### 4. Sliding windows detection algorithm

It's a method used in object detection, specifically for finding objects in images

- Start with a test image and pick a window size. You input a small rectangular region of the image into the ConvNet and it makes a prediction.
- Shift the window a little bit and input the new region into the ConvNet again. You keep doing this for every position in the image until you've covered the entire image.
- The hope is that if there's a car somewhere in the image, there will be a window where the ConvNet predicts that there's a car.
- By sliding the window across the image and classifying each region, you can detect the presence of a car.

*This method can be computationally expensive because you're running the ConvNet for many different regions.*

![sliding window](https://github.com/user-attachments/assets/50ae2904-b68d-4ad0-8282-d7d7ef53edfa)

#### 4.1 Convolutional implementation 

The algorithm uses 5 by 5 filters to scan the image and detect different features. These filters are like small windows that move across the image.

- Algorithm applies these filters to the image and maps it to a new size of 10 by 10 by 16. Then, it performs a process called max pooling to reduce the size to 5 by 5 by 16.
- Instead of using fully connected layers, the algorithm converts them into convolutional layers. It does this by using 5 by 5 filters with 400 channels. This means that each filter looks at all 16 channels of the previous layer. The output of this layer is a 1 by 1 by 400 volume.
- Algorithm continues with more convolutional layers using 1 by 1 filters. Finally, it uses a softmax activation function to output a 1 by 1 by 4 volume, representing the probabilities of different object classes.

*The convolutional implementation of sliding windows allows the algorithm to share computation and make predictions for the entire image at once, instead of running the algorithm multiple times for different regions of the image.*

![convolutional imple1](https://github.com/user-attachments/assets/21398b51-4a03-483a-8ad5-60c485db2d27)

![sliding window imple](https://github.com/user-attachments/assets/f4bdb27d-8296-44f9-b89c-e0a5a4185a83)

#### 4.2 Bounding box predictions

*YOLO -> You Only Look Once*

1. We start with an input image and divide it into a grid. For example, we can use a 3x3 grid.
2. We apply an image classification and localization algorithm to each grid cell. This algorithm helps us determine if there is an object in that cell and predicts the object's class and bounding box
3. For each grid cell, we create a label vector that contains information about the object in that cell. The label vector includes values like `PC` (presence of an object), `BX` (x-coordinate of the bounding box), `BY` (y-coordinate of the bounding box), `BH` (height of the bounding box), `BW` (width of the bounding box), and `class labels` for different object categories.
4. We train a neural network using these input images and target labels. The neural network learns to map the input image to the target output volume, which contains the predicted bounding boxes for each grid cell.
5. At test time, we feed an input image to the trained neural network and get the output volume. From this output, we can easily read off the presence of objects, their class labels, and the coordinates of their bounding boxes.

![Yolo](https://github.com/user-attachments/assets/d72541ff-d5f0-4a88-94d4-49fbe6dc3f33)

### 5. Intersection over union

Intersection over Union (IoU) is a function used to measure how well an object detection algorithm is working. When detecting objects, the algorithm is expected to not only identify the object but also accurately locate it within an image. 

You have the ground-truth bounding box (the actual location of the object) and the predicted bounding box (the location identified by the algorithm).IoU calculates the intersection and union of these two bounding boxes. The intersection is the area where the two boxes overlap, while the union is the total area covered by both boxes.
```
IoU = Intersection Area / Union Area
IoU > 1 -> Perfect
IoU > 0.5 -> Acceptable
```
*IoU is measure of the overlap between two bounding boxes*

![intersection over union](https://github.com/user-attachments/assets/d6ede558-5fe0-424a-9537-20c79eaa5061)

### 6. Non-Max Suppression

Non-max suppression is a technique used in object detection algorithms to ensure that each object is detected only once.

- The algorithm first looks at the probabilities associated with each detection. It selects the detection with the highest probability as the most confident detection. 
- Then, it compares the remaining detections to the most confident one. If any of the remaining detections have a high overlap with the most confident detection, they are suppressed or discarded.
- The algorithm repeats this process, selecting the next most confident detection and suppressing any overlapping detections, until all detections have been processed. 
- The final result is a set of non-overlapping detections, with each object represented by a single detection.

![non max supression](https://github.com/user-attachments/assets/99a6f492-8c36-454d-be46-dfa9727c67bb)

### 7. Anchor Box Algorithm

If you want to detect different objects in it, like cars, pedestrians, and motorcycles. The problem is that sometimes these objects can overlap or be very close to each other, making it hard for the computer to tell them apart.

To solve this, we use something called anchor boxes. Anchor boxes are pre-defined shapes that represent different types of objects.

- When the computer analyzes the image, it assigns each object to the grid cell that contains its center point. 
- But instead of just detecting one object per cell, it now associates each object with the anchor box that best matches its shape.
- This way, the computer can output multiple detections for each grid cell, allowing it to detect and classify different objects accurately.

![Anchor box](https://github.com/user-attachments/assets/a15069e2-ba8d-4c09-98b9-a051561cb535)

### 8. YOLO Algorithm

![yolo2](https://github.com/user-attachments/assets/256922d3-0b8b-4940-b528-5ccd2d427d60)

#### 8.1 Outputing non max supression

![outputin max sup](https://github.com/user-attachments/assets/b922ad83-f4f8-4153-b0d4-4e2b7146e2f4)