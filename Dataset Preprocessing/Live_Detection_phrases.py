import os
import sys
import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from imutils import face_utils
import tensorflow as tf

words = ['NULL', 'Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']
phrases = ['Stop Navigation', 'Excuse me', 'I am sorry', 'Thank you', 'Good bye', 'I love this game', 'Nice to meet you', 'You are welcome', 'How are you', 'Have a good time']


def face_extractor(img):
    try:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_classifier = cv2.CascadeClassifier('Dataset Preprocessing/xml files/haarcascade_frontalface_default.xml')
        faces = face_classifier.detectMultiScale(image, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cropped_image = image[y:y+h, x:x+w]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            return cropped_image
        else:
            print("No face found.")
            return None
    except Exception as e:
        print(f"Error in face_extractor: {e}")
        return None

def lips_extractor(img):
    try:
        predictor = dlib.shape_predictor('Dataset Preprocessing/xml files/shape_predictor_68_face_landmarks.dat')
        image = imutils.resize(img, width=56)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bbox = dlib.rectangle(0, 0, gray.shape[1], gray.shape[0])
        face_landmarks = predictor(gray, bbox)
        face_landmarks = face_utils.shape_to_np(face_landmarks)

        lip_image = None
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == 'mouth':
                (x, y, w, h) = cv2.boundingRect(np.array([face_landmarks[i:j]]))
                lip_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                lip_image = imutils.resize(lip_image, width=500, inter=cv2.INTER_CUBIC)
                lip_image = cv2.cvtColor(lip_image, cv2.COLOR_BGR2GRAY)
                break

        if lip_image is None or lip_image.size == 0:
            print("No lips detected.")
            return None
        else:
            return lip_image
    except Exception as e:
        print(f"Error in lips_extractor: {e}")
        return None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device")
    sys.exit()

frames = []
recording = False

print("Press Space to start/stop recording, and Esc to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  
        break
    elif key == 32:  
        recording = not recording
        if recording:
            print("Recording started.")
            frames = [] 
        else:
            print("Recording stopped.")

    if recording:
        frames.append(frame.copy())

cap.release()
cv2.destroyAllWindows()

print(f"Recorded {len(frames)} frames.")

# min and max frames
min_frames = 8
max_frames = 28
total_frames = len(frames)

if total_frames == 0:
    print("No frames recorded. Exiting.")
    sys.exit()

num_frames = min(max_frames, max(min_frames, total_frames // 10))
interval = max(1, total_frames // num_frames)

sequence = []
for i in range(num_frames):
    idx = i * interval
    if idx >= total_frames:
        idx = total_frames - 1
    frame = frames[idx]
    face = face_extractor(frame)
    if face is None:
        print(f"No face found in frame {idx}. Skipping frame.")
        continue
    lips = lips_extractor(face)
    if lips is None:
        print(f"No lips found in frame {idx}. Skipping frame.")
        continue
    resized_frame = resize(lips, (100, 100))
    resized_frame = (255 * resized_frame).astype(np.uint8)
    sequence.append(resized_frame)

print(f"Extracted {len(sequence)} frames.")

if len(sequence) == 0:
    print("No valid frames extracted. Exiting.")
    sys.exit()

# sequence padding
pad_length = 28 - len(sequence)
if pad_length > 0:
    pad_array = [np.zeros((100, 100), dtype=np.uint8)] * pad_length
    sequence.extend(pad_array)
sequence = np.array(sequence)


model_path = '/home/veeransh/Desktop/Project-X-Lip-Reading/Model Architecture/Saved Model/3D_CNN_Bi-LSTM.h5' # load model (change to your cnn+lstm or 3d cnn scratch (.h5 file))
try:
    loaded_model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# Normalizing
sequence = sequence.astype('float32') / 255.0  
sequence = sequence.reshape(1, 28, 100, 100, 1)  


try:
    ans = loaded_model.predict(sequence)
    percentages = [round(p * 100, 2) for p in ans[0]]
    predictions = {phrases[i]: percentages[i] for i in range(len(phrases))}
    
    for word, percent in predictions.items():
        print(f"{word}: {percent}%")
    
    max_index = np.argmax(ans)
    text = f"Predicted: {phrases[max_index]} with confidence {percentages[max_index]}%"
    print(text)
except Exception as e:
    print(f"Error during prediction: {e}")