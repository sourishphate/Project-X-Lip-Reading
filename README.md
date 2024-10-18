<h1 align="center"> Lip Reading 💬</h1>


<p align="center">
  <img src="https://github.com/user-attachments/assets/9052f2a1-c027-401e-a79f-e3835189365b" alt="read-my-lips" width="350" />
</p>


<div align="center">
  <strong>The goal is to convert visual information from lip movements into text. This involves recognizing and interpreting the movements of a speaker's lips to accurately transcribe spoken words.</strong>
</div>

<br />

## 📊 Results


## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Tech Stack](#️-tech-stack)
- [File Structure](#-file-structure)

## 📘 About the Project


## ⚙️ Tech Stack

| **Category**                | **Technologies**                                                                                       |
|-----------------------------|----------------------------------------------------------------------------------------------------|
| **Programming Languages**   | [![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)              |
| **Frameworks**              | [![TensorFlow](https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/) |
| **Libraries**               | [![OpenCV](https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)             |
| **Deep Learning Models**    | [![LSTM](https://img.shields.io/badge/LSTM-563D7C?style=for-the-badge&logo=lstm&logoColor=white)](https://en.wikipedia.org/wiki/Long_short-term_memory) [![CNN](https://img.shields.io/badge/CNN-0A192E?style=for-the-badge&logo=cnn&logoColor=white)](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/) |
| **Dataset**                 | [![MIRACL-VC1](https://img.shields.io/badge/MIRACL--VC1-4D2A4E?style=for-the-badge&logo=dataset&logoColor=white)](https://paperswithcode.com/dataset/miracl-vc1)                                                                            |
| **Tools**                   | [![Git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/) [![Google Colab](https://img.shields.io/badge/google%20colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)                            |
| **Visualization & Analysis**| [![Matplotlib](https://img.shields.io/badge/matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/seaborn-013243?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)                 |


## 📁 File Structure

    ├── Dataset Preprocessing
       ├── xml files
          ├── haarcascade_frontalface_default.xml
          ├── haarcascade_mcs_mouth.xml
          ├── shape_predictor_68_face_landmarks.dat
       ├── 01_Face_Extraction.ipynb
       ├── 02_Lip_Extraction.ipynb
       ├── 03_Train_Test_Split.ipynb
    ├── Hyperparameter Tuning
       ├── Grid Search.ipynb
       ├── Random Search.ipynb
    ├── Mini Projects
       ├── Cat_Dog_Classifier_CNN.ipynb
       ├── Human_Action_Recognition_LSTM.ipynb
       ├── Next_Word_Predictor_LSTM.ipynb
       ├── Video_Anomaly_Detection_CNN_LSTM.ipynb
    ├── Model Architecture
       ├── Saved Model
          ├── 3D_CNN_Bi-LSTM.h5
       ├── 3D_CNN.ipynb
       ├── 3D_CNN_Bi-LSTM.ipynb
       ├── 3D_CNN_From_Scratch.ipynb
       ├── 3D_CNN_LSTM.ipynb
       ├── Adam.ipynb
       ├── CategoricalCrossentropy.ipynb
       ├── Data_Augmentation.ipynb
       ├── Dropout.ipynb
       ├── EarlyStopping.ipynb
       ├── L1_Regularization.ipynb
       ├── L2_Regularization_1.ipynb
       ├── L2_Regularization_2.ipynb
       ├── RMSprop.ipynb
    ├── Model Evaluation
       ├── Accuracy.ipynb
       ├── Live_Detection.ipynb
       ├── Onlne_Testing.ipynb
       ├── Precision.ipynb
       ├── Recall.ipynb
    ├── Notes
       ├── LSTM
       ├── OpenCV
       ├── Om Mukherjee
       ├── Sourish Phate       
    ├── README.md

## 💾 Dataset: MIRACL-VC1

The **MIRACL-VC1** dataset is structured to facilitate research in visual speech recognition, particularly lip reading. Here's a breakdown of its structure and contents:

#### Data Composition:
- **Video Clips**: The dataset contains short video clips of multiple speakers reciting specific phrases. Each clip captures the upper body, focusing mainly on the face and mouth area.
- **Speakers**: It features several speakers from diverse backgrounds, which helps models generalize across different individuals and speaking styles.
- **Languages**: The dataset is typically in English, though speakers may vary in accents and pronunciations.
- **Phrases**: Each video clip corresponds to one of a predefined set of phrases, which are recited by the speakers. The phrases are usually short and may cover simple daily expressions or numbers.

#### Dataset Contains The Following Words and Phrases:

![image](https://github.com/user-attachments/assets/cc4d4ba3-0961-4439-8a36-0a0de73e2bac)

[Download the MIRACL-VC1 dataset on Kaggle](https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1)

