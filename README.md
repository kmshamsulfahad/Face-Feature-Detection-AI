#  Face Feature Detection AI

## Overview
This project performs real-time face feature detection using a webcam.  
It detects and classifies:

-  Left Eye (Open/Closed)
-  Right Eye (Open/Closed)
-  Mouth (Open/Closed)

---

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy

---

## How It Works
- Captures video using webcam
- Uses MediaPipe Face Mesh (468 landmarks)
- Calculates:
  - Eye Aspect Ratio (EAR)
  - Mouth Aspect Ratio (MAR)
- Classifies states based on thresholds

---

## Run the Project

```bash
pip install -r requirements.txt
python face_detection.py
