# collect_dataset.py
# Save the extracted features alongside a label

import numpy as np
import cv2  # opencv
import mediapipe as mp
import csv
import time
from feature_extraction import extract_features

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

# Dataset creation/addition
dataset_file = open('raw_dataset.csv', 'a', newline='')
writer = csv.writer(dataset_file)

# Open default camera (only change if more than one camera source)
capture = cv2.VideoCapture(0)
if not capture.isOpened():  # check camera open
    print('Cannot open camera')
    exit()

while True:
    frameRead, frame = capture.read()
    if not frameRead:
        print('Can\'t receive frame. Exiting...')
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # colour change back to BGR (openCV)
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    if ord('a') <= key <= ord('z'):

        # Don't record H or J yet
        if key == ord('h') or key == ord('j'):
            print('H OR J PRESSED, EXITING')
            break
        
        label = chr(key).upper()
        print('GET READY, HOLD UNTIL PROGRAM SAYS IT HAS BEEN SAVED')
        time.sleep(3)

        for i in range (30):
            features = extract_features(hands, frame)
            writer.writerow(label + features)
        print(f'Letter {label} has been saved 30 times')

capture.release()
cv2.destroyAllWindows()