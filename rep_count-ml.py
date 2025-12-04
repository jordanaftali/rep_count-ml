import cv2
import mediapipe as mp
import numpy as np
import os

# ========= CONFIG =========
# Change this to the video you want to use:
# e.g. "lateral_raise_shoulder_dumbells_beginner_lateral_view.mp4"
#      "curls_presses_dumbells_beginner_lateral_view.mp4"
VIDEO_FILENAME = "curls_presses_dumbells_beginner_front_view.mp4"

# ========= PATH SETUP =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "data", "videos")
VIDEO_PATH = os.path.join(VIDEOS_DIR, VIDEO_FILENAME)

print("VIDEO_PATH:", VIDEO_PATH)
print("Exists?:", os.path.exists(VIDEO_PATH))

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# ========= ANGLE FUNCTION =========
def calculate_angle(a, b, c):
    """Return the angle (deg) at point b for points a-b-c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


# ========= MAIN REP COUNTER =========
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: could not open video:", VIDEO_PATH)
        return

    reps = 0
    arm_up = False          # are we currently in the "up" phase?

    # we will learn the angle range over time and set thresholds from it
    min_angle = None
    max_angle = None

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            angle = None
            up_threshold = None
            down_threshold = None

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Use RIGHT arm: SHOULDER → ELBOW → WRIST
                shoulder = [
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                ]
                elbow = [
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                ]
                wrist = [
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                ]

                angle = calculate_angle(shoulder, elbow, wrist)

                # ---- learn min / max angle dynamically ----
                if min_angle is None:
                    min_angle = angle
                    max_angle = angle
                else:
                    min_angle = min(min_angle, angle)
                    max_angle = max(max_angle, angle)

                angle_range = max_angle - min_angle

                # only start counting reps once we see enough movement
                if angle_range > 15.0:
                    # thresholds at 30% and 70% of the observed range
                    up_threshold = min_angle + 0.3 * angle_range
                    down_threshold = min_angle + 0.7 * angle_range

                    # A rep = angle goes UP (below up_threshold)
                    #        then comes back DOWN (above down_threshold)
                    if angle < up_threshold and not arm_up:
                        arm_up = True
                    elif angle > down_threshold and arm_up:
                        arm_up = False
                        reps += 1
                        print(f"REP! reps={reps}, angle={angle:.1f}")

                # draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                # ========= OVERLAYS =========
            # Reps
            cv2.putText(
                frame,
                f"Reps: {reps}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 0),
                4,
            )

            # Angle
            if angle is not None:
                cv2.putText(
                    frame,
                    f"Angle: {int(angle)}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                )

            # Debug: show min / max and thresholds so you can see what's happening
            if min_angle is not None and max_angle is not None:
                cv2.putText(
                    frame,
                    f"min:{min_angle:.1f} max:{max_angle:.1f}",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )
            if up_threshold is not None and down_threshold is not None:
                cv2.putText(
                    frame,
                    f"up:{up_threshold:.1f} down:{down_threshold:.1f}",
                    (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Rep Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
