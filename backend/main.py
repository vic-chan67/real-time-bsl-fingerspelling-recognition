import numpy as np
import cv2 as cv

capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    # ret: frame successfully read (True/False)
    ret, frame = capture.read()

    # If frame read correctly, ret = True
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Operations

    # Display resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
capture.release()
cv.destroyAllWindows()