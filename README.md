# rep_count-ml

A small experiment in using computer vision and MediaPipe to count strength training reps from regular videos.

This project was originally built for a live presentation at PyLadies ATX.

As a product designer, this project was also an opportunity to explore interface possibilities, user flows, and vision tools can support better training experiences.

---

## The Challenge

- There is no simple or affordable way to automatically count repetitions from regular workout videos.  
- People, trainers, and coaches often rely on manual counting, which is slow and prone to human error.  
- When training alone, it is difficult to track proper form or monitor progress consistently.

---

## The Approach and Solution

The solution uses MediaPipe Pose to detect body landmarks frame-by-frame, compute joint angles, and determine when a full repetition has occurred.

The development flow followed three steps:

1. **Exploration**  
   Use `view_landmarks_video.py` to confirm that MediaPipe Pose is working and that the skeleton appears correctly over each frame.

2. **Data and Debugging (optional)**  
   Use `extract_keypoints.py` to export landmarks and angles to CSV in order to inspect the raw values and understand the movement better.

3. **Rep Counting**  
   Use `rep_counter-ml.py` to apply angle thresholds and detect full repetitions in real time based on the video input.

---
rep_count-ml/
│
├── data/
│ └── videos/
│ ├── curls_presses_dumbells_....mp4
│ ├── lateral_raise_shoulder_....mp4
│ ├── ...
│ └── *_keypoints.csv
├── extract_keypoints.py
├── view_landmarks_video.py
├── rep_counter-ml.py
├── README.md


## Presentation Slides

Slides used during the PyLadies ATX presentation:
[rep-count.pdf](slides/rep-count.pdf)

