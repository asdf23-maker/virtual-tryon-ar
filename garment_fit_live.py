import argparse, os, time
import cv2
import numpy as np
import mediapipe as mp

# --------------------------
# Helpers
# --------------------------
def ensure_bgra(img):
    """Make sure image is BGRA (has alpha)."""
    if img is None:
        return None
    if img.shape[2] == 4:
        return img
    bgr = img
    alpha = np.full((bgr.shape[0], bgr.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([bgr, alpha], axis=2)

def rotate_image_bgra(img_bgra, angle_deg, scale=1.0):
    """Rotate a BGRA image around its center."""
    h, w = img_bgra.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, scale)
    rotated = cv2.warpAffine(img_bgra, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    # Keep 4 channels
    if rotated.shape[2] == 3:
        rotated = ensure_bgra(rotated)
    return rotated

def resize_keep_aspect(img_bgra, new_w=None, new_h=None):
    h, w = img_bgra.shape[:2]
    if new_w is None and new_h is None:
        return img_bgra
    if new_w is not None and new_h is None:
        r = new_w / float(w)
        return cv2.resize(img_bgra, (int(w*r), int(h*r)), interpolation=cv2.INTER_LINEAR)
    if new_h is not None and new_w is None:
        r = new_h / float(h)
        return cv2.resize(img_bgra, (int(w*r), int(h*r)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(img_bgra, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def alpha_blend_bgra(under_bgr, overlay_bgra, x, y):
    """Alpha blend overlay_bgra onto under_bgr at (x, y). Modifies under_bgr in place."""
    oh, ow = overlay_bgra.shape[:2]
    uh, uw = under_bgr.shape[:2]
    # Clip ROI
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(uw, x + ow), min(uh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return
    overlay_crop = overlay_bgra[(y1 - y):(y2 - y), (x1 - x):(x2 - x), :]
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    under_roi = under_bgr[y1:y2, x1:x2, :].astype(np.float32)
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    blended = alpha * overlay_rgb + (1 - alpha) * under_roi
    under_bgr[y1:y2, x1:x2, :] = blended.astype(np.uint8)

def draw_soft_shadow(frame, mask_bgra, x, y, blur=13, offset=(10, 18), strength=0.35):
    """Simple soft shadow based on the garment alpha."""
    oh, ow = mask_bgra.shape[:2]
    uh, uw = frame.shape[:2]
    # Create shadow image
    alpha = (mask_bgra[:, :, 3] / 255.0 * 255).astype(np.uint8)
    shadow = np.zeros((oh, ow), dtype=np.uint8)
    shadow[:] = 0
    shadow = np.maximum(shadow, alpha)
    shadow = cv2.GaussianBlur(shadow, (blur | 1, blur | 1), 0)
    # Place shadow on frame with offset
    sx = x + offset[0]
    sy = y + offset[1]
    x1, y1 = max(0, sx), max(0, sy)
    x2, y2 = min(uw, sx + ow), min(uh, sy + oh)
    if x1 >= x2 or y1 >= y2:
        return
    crop = shadow[(y1 - sy):(y2 - sy), (x1 - sx):(x2 - sx)].astype(np.float32) / 255.0
    # Darken frame under the shadow
    roi = frame[y1:y2, x1:x2, :].astype(np.float32)
    darkened = roi * (1.0 - crop * strength)
    frame[y1:y2, x1:x2, :] = darkened.astype(np.uint8)

class KeypointSmoother:
    """Exponential moving average (EMA) smoother for jitter-free landmarks."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None

    def __call__(self, pts):
        pts = np.array(pts, dtype=np.float32)
        if self.prev is None:
            self.prev = pts
            return pts
        self.prev = self.alpha * pts + (1 - self.alpha) * self.prev
        return self.prev

# --------------------------
# Main live try-on
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Live VTO with garment warping (MediaPipe Pose).")
    ap.add_argument("--garment", required=True, help="Path to transparent garment PNG (BGRA).")
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0).")
    ap.add_argument("--scale", type=float, default=1.35, help="Base width scale vs shoulder width.")
    ap.add_argument("--ema", type=float, default=0.6, help="EMA smoothing factor (0-1, higher = smoother).")
    ap.add_argument("--flip", action="store_true", help="Flip camera horizontally (selfie).")
    ap.add_argument("--show-pose", action="store_true", help="Draw pose for debugging.")
    ap.add_argument("--save-dir", default="outputs", help="Directory to save screenshots.")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    garment = cv2.imread(args.garment, cv2.IMREAD_UNCHANGED)
    if garment is None:
        raise FileNotFoundError(f"Garment not found: {args.garment}")
    garment = ensure_bgra(garment)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    smoother = KeypointSmoother(alpha=args.ema)

    # Keyboard-adjustable fine-tuning
    scale_k = args.scale
    x_nudge, y_nudge = 0, 0

    print("Controls: q=quit | s=screenshot | +/- scale | arrows move | p toggle pose")
    show_pose = args.show_pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                def lp(i):
                    return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float32)

                L_SH = lp(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                R_SH = lp(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                L_HP = lp(mp_pose.PoseLandmark.LEFT_HIP.value)
                R_HP = lp(mp_pose.PoseLandmark.RIGHT_HIP.value)

                # Smooth landmarks (shoulders + hips)
                smoothed = smoother([L_SH, R_SH, L_HP, R_HP])
                L_SH, R_SH, L_HP, R_HP = smoothed

                # Geometry
                shoulder_vec = R_SH - L_SH
                shoulder_width = np.linalg.norm(shoulder_vec)
                if shoulder_width < 5:  # too small / far
                    cv2.putText(frame, "Move closer", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow("VTO Live", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                shoulder_center = (L_SH + R_SH) / 2.0
                # Angle in degrees (negative to match OpenCV rotation direction)
                angle = -np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))

                # Target width for garment
                target_w = max(1, int(shoulder_width * scale_k))
                g_h, g_w = garment.shape[:2]
                aspect = g_h / g_w if g_w else 1.0
                target_h = max(1, int(target_w * aspect))

                g_resized = resize_keep_aspect(garment, new_w=target_w)
                g_rot = rotate_image_bgra(g_resized, angle)

                # Optional: slight vertical bias to start above shoulder center
                # Position top-left so garment center aligns near shoulder_center with nudges
                gh, gw = g_rot.shape[:2]
                center_x = int(shoulder_center[0]) + x_nudge
                center_y = int(shoulder_center[1]) + y_nudge - int(0.15 * gh)  # lift a bit
                top_left_x = center_x - gw // 2
                top_left_y = center_y - gh // 4

                # Soft shadow under garment for depth
                draw_soft_shadow(frame, g_rot, top_left_x, top_left_y, blur=21, offset=(12, 20), strength=0.30)
                # Composite
                alpha_blend_bgra(frame, g_rot, top_left_x, top_left_y)

                if show_pose:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                cv2.putText(frame, "No pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # HUD
            cv2.putText(frame, f"scale:{scale_k:.2f}  ema:{smoother.alpha:.2f}", (20, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("VTO Live", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                path = os.path.join(args.save_dir, f"tryon_{ts}.png")
                cv2.imwrite(path, frame)
                print(f"Saved screenshot: {path}")
            elif key in (ord('+'), ord('=')):
                scale_k = min(3.0, scale_k + 0.05)
            elif key == ord('-'):
                scale_k = max(0.6, scale_k - 0.05)
            elif key == 81:   # left arrow
                x_nudge -= 5
            elif key == 83:   # right arrow
                x_nudge += 5
            elif key == 82:   # up arrow
                y_nudge -= 5
            elif key == 84:   # down arrow
                y_nudge += 5
            elif key == ord('p'):
                show_pose = not show_pose

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
