import cv2
import mediapipe as mp
import numpy as np
import os

# Load garment image
garment_path = "garment.png"
print("📁 Current working directory:", os.getcwd())
print("📄 Looking for garment at:", os.path.join(os.getcwd(), garment_path))

garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
if garment is None:
    raise FileNotFoundError(f"❌ Unable to load garment image. Make sure '{garment_path}' exists in the folder: {os.getcwd()}")
print(f"✅ Image loaded! Shape: {garment.shape}")

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ No webcam feed – check camera connection or permissions.")
print("✅ Webcam started")

# Function to overlay image with alpha channel
def overlay_image_alpha(background, overlay, x, y):
    h, w = overlay.shape[:2]
    b_h, b_w = background.shape[:2]

    if x < 0: overlay = overlay[:, -x:]; w += x; x = 0
    if y < 0: overlay = overlay[-y:, :]; h += y; y = 0
    if x+w > b_w: overlay = overlay[:, :b_w-x]; w = b_w-x
    if y+h > b_h: overlay = overlay[:b_h-y, :]; h = b_h-y

    if h <= 0 or w <= 0:
        return background

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_img

    return background

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        h, w, _ = frame.shape
        x1 = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
        y1 = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        x2 = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
        y2 = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)

        # Draw shoulders for debug
        cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)

        center_x = (x1 + x2) // 2
        width = abs(x2 - x1) * 2
        new_width = int(width)
        new_height = int(garment.shape[0] * (new_width / garment.shape[1]))

        resized_garment = cv2.resize(garment, (new_width, new_height), interpolation=cv2.INTER_AREA)

        top_left_x = center_x - new_width // 2
        top_left_y = int((y1 + y2) // 2)

        # Debug position
        print(f"🧥 Garment position: x={top_left_x}, y={top_left_y}, w={new_width}, h={new_height}")
        cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + new_width, top_left_y + new_height), (0, 0, 255), 2)

        # Overlay
        frame = overlay_image_alpha(frame, resized_garment, top_left_x, top_left_y)

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
