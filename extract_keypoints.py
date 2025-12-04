import cv2
import mediapipe as mp

VIDEO_INPUT = "data/lateral_raise_shoulder_dumbbells_beginner_lateral_view.mp4"
VIDEO_OUTPUT = "data/lateral_raise_landmarks_output.mp4"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_landmarks(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Save output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

            out.write(frame)
    
    cap.release()
    out.release()
    print(f"SUCCESS! Video saved to: {output_path}")


if __name__ == "__main__":
    draw_landmarks(VIDEO_INPUT, VIDEO_OUTPUT)
