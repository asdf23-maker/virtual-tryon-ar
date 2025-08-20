import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def fit_garment(user_img_path, garment_img_path, output_path="outputs/fitted_output.png"):
    # Load images
    user_img = cv2.imread(user_img_path)
    garment_img = cv2.imread(garment_img_path, cv2.IMREAD_UNCHANGED)  # keep transparency if PNG

    if user_img is None or garment_img is None:
        print("❌ Error: Image not found.")
        return

    # Detect pose
    results = pose.process(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        print("❌ No person detected in user image.")
        return

    h, w, _ = user_img.shape

    # Example: get shoulder points
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    x1, y1 = int(left_shoulder.x * w), int(left_shoulder.y * h)
    x2, y2 = int(right_shoulder.x * w), int(right_shoulder.y * h)

    shoulder_width = abs(x2 - x1)
    garment_resized = cv2.resize(garment_img, (shoulder_width * 2, int(shoulder_width * 2.5)))

    # Overlay garment on top of user
    y_offset = y1 - garment_resized.shape[0] // 4
    x_offset = min(x1, x2) - garment_resized.shape[1] // 4

    result = user_img.copy()

    # Paste garment (simple overlay)
    for c in range(0, 3):
        y1_overlay = max(0, y_offset)
        y2_overlay = min(result.shape[0], y_offset + garment_resized.shape[0])
        x1_overlay = max(0, x_offset)
        x2_overlay = min(result.shape[1], x_offset + garment_resized.shape[1])

        result[y1_overlay:y2_overlay, x1_overlay:x2_overlay, c] = garment_resized[
            0:(y2_overlay - y1_overlay),
            0:(x2_overlay - x1_overlay),
            c
        ]

    # Ensure output folder exists
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"✅ Garment fitted image saved at: {output_path}")


# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    if len(sys.argv) == 3:
        fit_garment(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python garment_fit.py user_fullbody.png garment.png")
