import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Paths
MODEL_PATH = r"C:\Users\surya\projects\sign_language_detector\models\best_model.keras"
CLASSES_PATH = r"C:\Users\surya\projects\sign_language_detector\data\processed\classes.txt"

# Load model and labels
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully")
with open(CLASSES_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(f"Model loaded. Classes: {classes}")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Parameters
SEQ_LEN = 60
FEATURE_DIM = 126
buffer = []
prediction = "Waiting..."
confidence = 0.0
prev_time = time.time()

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
print("Camera opened. Press 'q' to quit.")

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    data = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])
    while len(data) < FEATURE_DIM:
        data.append(0.0)
    return np.array(data[:FEATURE_DIM])

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    # Extract landmarks
    landmarks = extract_landmarks(frame)
    buffer.append(landmarks)
    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    # Predict when buffer is full
    if len(buffer) == SEQ_LEN:
        input_seq = np.array(buffer).reshape(1, SEQ_LEN, FEATURE_DIM)
        preds = model.predict(input_seq, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]
        if conf > 0.7:
            prediction = classes[idx]
            confidence = conf
        else:
            prediction = "???"
            confidence = conf

    # Display
    cv2.putText(frame, f"Sign: {prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("ISL Real-Time Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()