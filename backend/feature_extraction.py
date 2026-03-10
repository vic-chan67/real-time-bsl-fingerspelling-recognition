# feature_extraction.py
# Draw hand landmarks on static image and display x,y,z coords (todo: normalise coords against wrist coords)

import numpy as np
import cv2  # opencv
import mediapipe as mp
import math
# from landmark_names import landmark_names

def extract_features(hands, frame):

    # Define all feature vectors
    feature_vector = []
    right_hand_vector = []
    left_hand_vector = []

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)     # flip image and change colour from BGR to RGB

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[i].classification[0].label    # get hand labels
            landmarks = hand_landmarks.landmark     # get landmarks

            # Get WRIST and MIDDLE_FINGER_TIP coords
            wrist = landmarks[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
            middleTip = landmarks[12]
            middleTip_x, middleTip_y, middleTip_z = middleTip.x, middleTip.y, middleTip.z
            
            scale = math.sqrt((middleTip_x-wrist_x)**2 + 
                              (middleTip_y-wrist_y)**2 + 
                              (middleTip_z-wrist_z)**2)     # calculate scale value

            for landmark in landmarks:      # normalisation
                rel_x = landmark.x - wrist_x    # calculate coords relative to WRIST
                rel_y = landmark.y - wrist_y
                rel_z = landmark.z - wrist_z

                scaled_x = rel_x / scale    # calculate scaled coords
                scaled_y = rel_y / scale
                scaled_z = rel_z / scale

                # Save each hand's coords into it's own vector
                if hand_label == 'Right':
                    right_hand_vector.extend([scaled_x, scaled_y, scaled_z])
                else:
                    left_hand_vector.extend([scaled_x, scaled_y, scaled_z])

        # Deal with missing hand(s) coords
        if len(right_hand_vector) == 0:
            right_hand_vector = [0] * 63
        if len(left_hand_vector) == 0:
            left_hand_vector = [0] * 63
        feature_vector = right_hand_vector + left_hand_vector   # combine vectors
        # print(len(feature_vector))    # 21 landmarks * 3 coords * 2 hands = 126 values
    
    return feature_vector

# File executable
if __name__ == '__main__':
    # ../assets/woman_hands.jpg from https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg
    image = cv2.imread('/Users/dev/Documents/real-time-bsl-fingerspelling-recognition/assets/woman_hands.jpg') 
    if image is None:
        print("Image failed to load")
        quit()
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

    print(len(extract_features(hands, image1)))