# Real-Time BSL FingerSpelling Recognition Software

Real-time recognition of British Sign Language (BSL) finger spelling using computer vision and machine learning techniques. Aims to help BSL learners make sure their finger spelling is correct without requiring a third-party person to confirm, allowing learning from home with no personal tutor. Real-time recognition ensures immediate feedback without delays to speed up learning. 

## Setup
The program is designed to be fully executable at any stage in the plan, whether that be just the camera working or displaying the full predictions in real-time.

### Instructions
1. Clone repository
2. Activate virtual environment - **macOS/Linux:** `source backend/.venv/bin/activate` | **Windows (cmd prompt):** `backend\.venv\Scripts\activate.bat`
3. Install dependencies: `pip install -r requirements.txt`
4. Run **main.py**

## Plan
- Phase 1:
  - Capture camera input ✅
  - Detect hand landmarks ✅
  - Extract and visualise features
- Phase 2:
  - Build dataset
- Phase 3:
  - Build/train classifier on dataset
- Phase 4:
  - Pass live features through classifier
  - Display predictions in real-time

## Architecture (not set in stone)
- Python backend
  - MediaPipe for hand landmark detection
  - Classifier for prediction
  - REST API for real-time
- Frontend in browser
  - Webcam capture
  - Send frames to backend
  - Display real-time prediction
