import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import os

# Paths
person_img_path = 'uploads/person.jpg'
garment_img_path = 'uploads/garment.png'
output_path = 'outputs/tryon_result.png'

# Load images
person_img = cv2.imread(person_img_path)
garment_img_pil = Image.open(garment_img_path).convert("RGBA")

if person_img is None:
    raise FileNotFoundError(f"❌ Person image not found at {person_img_path}")
if garment_img_pil is None:
    raise FileNotFoundError(f"❌ Garment image not found at {garment_img_path}")

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
results = pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))

if not results.pose_landmarks:
    raise RuntimeError("❌ No pose landmarks detected.")

# Get shoulders
landmarks = results.pose_landmarks.landmark
left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

h, w, _ = person_img.shape
lx, ly = int(left_shoulder.x * w), int(left_shoulder.y * h)
rx, ry = int(right_shoulder.x * w), int(right_shoulder.y * h)

# Compute center & width
center_x = (lx + rx) // 2
shoulder_width = int(np.linalg.norm([rx - lx, ry - ly]) * 1.4)

# Resize garment
garment_resized = garment_img_pil.resize((shoulder_width, int(shoulder_width * garment_img_pil.height / garment_img_pil.width)))
gx, gy = center_x - garment_resized.width // 2, min(ly, ry)

# Convert person image to PIL
person_img_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
person_img_pil.paste(garment_resized, (gx, gy), garment_resized)

# Save final try-on image
os.makedirs('outputs', exist_ok=True)
person_img_pil.save(output_path)
print(f"✅ Garment fitted successfully. Output saved to {output_path}")
