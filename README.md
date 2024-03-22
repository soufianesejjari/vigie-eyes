## Vigie-Eyes
Vigie-Eyes is a visual monitoring system designed for real-time fatigue detection. This project was carried out as part of the Multimedia Processing and Analysis module,  during the academic year 2023-2024, as part of the Master 2IAD program in the Computer Science department.

## Prerequisites
- Python 3.x
- Libraries: scipy, imutils, numpy, dlib, OpenCV
## Usage
- Download or clone the repository: git clone https://github.com/soufianesejjari/vigie-eyes
- Ensure required libraries are installed:  ``` pip install -r requirements.txt ```
- Run the Python script: ``` python vigie_eyes.py --shape-predictor shape_predictor_68_face_landmarks.dat --video video.mp4 ```
## Description
- This project utilizes the Eye Aspect Ratio (EAR) analysis for detecting signs of fatigue. The code continuously analyzes the video stream, detects faces, and computes the EAR for each eye. It triggers an alert if the eyes remain closed for a specific period, indicating a risk of eye fatigue.

## Features
- Real-time detection of faces and eyes
- Continuous monitoring of Eye Aspect Ratio (EAR)
- Alerts for potential eye fatigue after a specific period of eye closure
- For more details on the implementation and system operation, please refer to the vigie_eyes.py file.
