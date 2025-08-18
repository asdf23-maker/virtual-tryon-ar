import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image_path, output_path="pose_output.jpg"):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(output_path, image)
            print(f"Pose saved at {output_path}")
        else:
            print("No pose detected!")

# Example
# detect_pose("user_fullbody.png")
def get_keypoints(image_path):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark
            keypoints = {
                "left_shoulder": (int(landmarks[11].x * w), int(landmarks[11].y * h)),
                "right_shoulder": (int(landmarks[12].x * w), int(landmarks[12].y * h)),
                "left_hip": (int(landmarks[23].x * w), int(landmarks[23].y * h)),
                "right_hip": (int(landmarks[24].x * w), int(landmarks[24].y * h)),
            }
            return keypoints
        else:
            return None

