import cv2
import mediapipe as mp
import numpy as np
import os

print("📁 Current working directory:", os.getcwd())
garment_path = "garment.png"

# Load garment with alpha
garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
if garment is None or garment.shape[2] != 4:
    raise FileNotFoundError(f"❌ Unable to load garment image. Make sure '{garment_path}' is a transparent PNG and in folder: {os.getcwd()}")

print(f"✅ Garment loaded with shape: {garment.shape}")
gh, gw = garment.shape[:2]

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible – check camera or index")

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        pts_dst = np.float32([
            [lm[11].x * w, lm[11].y * h],  # Left shoulder
            [lm[12].x * w, lm[12].y * h],  # Right shoulder
            [lm[23].x * w, lm[23].y * h],  # Left hip
            [lm[24].x * w, lm[24].y * h],  # Right hip
        ])

        # Optional: Draw landmarks for debug
        for pt in pts_dst:
            cv2.circle(frame, tuple(np.int32(pt)), 5, (0, 255, 0), -1)

        # Garment source points
        pts_src = np.float32([[0, 0], [gw, 0], [0, gh], [gw, gh]])

        # Warp garment
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(garment, M, (w, h))

        # Alpha blend
        alpha = warped[..., 3] / 255.0
        for c in range(3):
            frame[..., c] = (1 - alpha) * frame[..., c] + alpha * warped[..., c]

    cv2.imshow("🧥 Real-Time Virtual Try-On", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
