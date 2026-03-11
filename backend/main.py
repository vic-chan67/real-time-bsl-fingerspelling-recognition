# main.py
# Full pipeline

import numpy as np
import cv2  # opencv
import mediapipe as mp

# .hands: hand detection/tracking
# .drawing_utils: draw landmarks and connections
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Hand detector instance (2 hands needed for most fingerspelling)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open default camera (only change if more than one camera source)
capture = cv2.VideoCapture(0)
if not capture.isOpened():  # check camera open
    print('Cannot open camera')
    exit()

while True:
    frameRead, frame = capture.read()
    if not frameRead:
        print('Cannot receive frame. Exiting...')
        break
    
    # Preprocessing:
    # Flip image (mirrors hands), colour change BGR (openCV) to RGB (mediapipe)
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False  # stop numpy copying image

    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # colour change back to BGR (openCV)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display
    cv2.imshow('Show hand landmarks', image)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()