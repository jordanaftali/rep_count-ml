import cv2
import mediapipe as mp
import os

# pasta onde está este arquivo .py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# pasta dos vídeos
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "data", "videos")


print("SCRIPT_DIR:", SCRIPT_DIR)
print("VIDEOS_DIR:", VIDEOS_DIR)
print("FILES IN VIDEOS_DIR:", os.listdir(VIDEOS_DIR))


VIDEO_FILENAME = "curls_presses_dumbells_beginner_front_view.mp4"
VIDEO_PATH = os.path.join(VIDEOS_DIR, VIDEO_FILENAME)

print("VIDEO_PATH:", VIDEO_PATH)
print("Exists?:", os.path.exists(VIDEO_PATH))

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: could not open video.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # fim do vídeo

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            cv2.imshow("Landmarks", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
