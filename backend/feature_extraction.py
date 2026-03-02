# feature_extraction.py
# Draw hand landmarks on static image and display x,y,z coords (todo: normalise coords against wrist coords)

import numpy as np
import cv2  # opencv
import mediapipe as mp

# woman_hands.jpg from https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg
image = cv2.imread('/Users/dev/Documents/real-time-bsl-fingerspelling-recognition/backend/woman_hands.jpg')
if image is None:
    print("Image failed to load")
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# .hands: hand detection/tracking
# .drawing_utils: draw landmarks and connections
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
 
# Hand detector instance (2 hands needed for most fingerspelling)
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

results = hands.process(image1)

# Draw hand landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark
        print(f"Number of landmarks: {len(landmarks)}")
        