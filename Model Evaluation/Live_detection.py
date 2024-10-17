import numpy as np
import os
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize
import imageio
from imutils import face_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

words = ['NULL', 'Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']

def face_extractor(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_classifier = cv2.CascadeClassifier(r'D:\Project-X-Lip-Reading\Dataset Preprocessing\xml files\haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(image, 1.3, 5)

    # If faces are found, extract the first face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cropped_image = image[y:y+h, x:x+w]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            return cropped_image
    else:
        print("No face found.")
        return None
    
def lips_extractor(img):
    predictor = dlib.shape_predictor(r'D:\Project-X-Lip-Reading\Dataset Preprocessing\xml files\shape_predictor_68_face_landmarks.dat')

    image = imutils.resize(img, width=56)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bbox = dlib.rectangle(0, 0, gray.shape[1], gray.shape[0])
    face_landmarks = predictor(gray, bbox)
    face_landmarks = face_utils.shape_to_np(face_landmarks)

    for (name,(i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
         if name=='mouth':
            for (x, y) in face_landmarks[i:j]:
                (x, y, w, h) = cv2.boundingRect(np.array([face_landmarks[i:j]]))
                lip_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                lip_image = imutils.resize(lip_image, width=500, inter=cv2.INTER_CUBIC)

                lip_image = cv2.cvtColor(lip_image, cv2.COLOR_BGR2GRAY)

    if len(lip_image) == 0:
        print("No lips detected.")
        return None
    else:
        return lip_image
    
cap = cv2.VideoCapture(0)
frames = []
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1)
    if key == 27:  # Escape key to exit
        break
    elif key == 32:  # Spacebar to start/stop recording
        recording = not recording

    if recording:
        frames.append(frame)
        cv2.putText(frame, "Recording...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Press Space to Record", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Video", frame)

cap.release()
cv2.destroyAllWindows()

print(f"Recorded {len(frames)} frames.")

min_frames = 8
max_frames = 28
total_frames = len(frames)

# Calculate the number of frames to extract
num_frames = min(max_frames, max(min_frames, total_frames // 10))

# Calculate the interval between frames
interval = total_frames // num_frames

sequence = []
for i in range(num_frames):
    frame = frames[i * interval]
    frame = face_extractor(frame)
    frame = lips_extractor(frame)
    frame = resize(frame, (100,100))
    frame = 255 * frame
    frame = frame.astype(np.uint8)
    sequence.append(frame)

print(f"Extracted {len(sequence)} frames.")

pad_array = [np.zeros((100, 100))]
sequence.extend(pad_array * (28 - len(sequence)))
sequence = np.array(sequence)

import tensorflow as tf
# Load the model
loaded_model = tf.keras.models.load_model('D:\Project-X-Lip-Reading\Model Architecture\Saved Model\\3D_CNN_LSTM_words_2_10.h5')

# Normalize the sequence
np.seterr(divide='ignore', invalid='ignore')  # Ignore divide by 0 warning
v_min = sequence.min(axis=(1, 2), keepdims=True)
v_max = sequence.max(axis=(1, 2), keepdims=True)
sequence = (sequence - v_min) / (v_max - v_min)
sequence = np.nan_to_num(sequence)

# Reshape the input for prediction
my_pred = sequence.reshape(1, 28, 100, 100, 1)
ans = loaded_model.predict(my_pred)

# Get all words with their percentages
percentages = [round(p * 100, 2) for p in ans[0]]
predictions = {words[i]: percentages[i] for i in range(len(words))}

# Print all words with their percentages
for word, percent in predictions.items():


    print(f"Predicted: {word} , {percent} %")

max_index = np.argmax(ans)
text = f"Spoken words is {words[max_index]} with the confidence of {percentages[max_index]} %"

print(text)

def display_images(image_list):
    fig, axes = plt.subplots(2, 14, figsize=(14, 2))
    for i, img in enumerate(image_list):
        row, col = divmod(i, 14)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')  # Hide axes
    plt.show()

display_images(sequence)