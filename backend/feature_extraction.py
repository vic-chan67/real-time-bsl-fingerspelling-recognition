# feature_extraction.py
# Draw hand landmarks on static image and display x,y,z coords (todo: normalise coords against wrist coords)

import numpy as np
import cv2  # opencv
import mediapipe as mp
import math
from landmark_names import landmark_names

# ../assets/woman_hands.jpg from https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg
image = cv2.imread('/Users/dev/Documents/real-time-bsl-fingerspelling-recognition/assets/woman_hands.jpg')
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

feature_vector = []

# Draw hand landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = hand_landmarks.landmark

        # Get WRIST and MIDDLE_FINGER_TIP coords
        wrist = landmarks[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
        middleTip = landmarks[12]
        middleTip_x, middleTip_y, middleTip_z = middleTip.x, middleTip.y, middleTip.z
        
        # 3D Euclidean distance (aka scale value)
        scale = math.sqrt((middleTip_x-wrist_x)**2 + (middleTip_y-wrist_y)**2 + (middleTip_z-wrist_z)**2)

        # Normalisation
        for i, landmark in enumerate(landmarks):
            # Get all coords relative to WRIST
            rel_x = landmark.x - wrist_x
            rel_y = landmark.y - wrist_y
            rel_z = landmark.z - wrist_z

            scaled_x = rel_x / scale
            scaled_y = rel_y / scale
            scaled_z = rel_z / scale

            feature_vector.append(scaled_x)
            feature_vector.append(scaled_y)
            feature_vector.append(scaled_z)
    
    # Check feature vector length:
    # 21 landmarks * 3 coords * 2 hands = 126 values
    # 21 landmarks * 3 coords * 1 hand = 63 values
    print(len(feature_vector))