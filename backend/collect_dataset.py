# collect_dataset.py
# Save the extracted features alongside a label

import numpy as np
import cv2  # opencv
import mediapipe as mp
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

# Open default camera (only change if more than one camera source)
capture = cv2.VideoCapture(0)
if not capture.isOpened():  # check camera open
    print("Cannot open camera")
    exit()

while True:
    frameRead, frame = capture.read()
    if not frameRead:
        print("Can't receive frame. Exiting...")
        break
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # colour change back to BGR (openCV)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()