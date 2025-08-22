import cv2
import numpy as np

def overlay_cloth(frame, cloth):
    # If cloth has alpha channel, use it; else build mask from intensity
    if cloth.shape[2] == 4:
        b,g,r,a = cv2.split(cloth)
        rgb = cv2.merge([b,g,r])
        alpha = a
    else:
        rgb = cloth
        gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    alpha_f = alpha.astype(float)/255.0
    alpha_b = 1.0 - alpha_f

    # Ensure same size
    h, w = frame.shape[:2]
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h))
        alpha_f = cv2.resize(alpha_f, (w, h))
        alpha_b = 1.0 - alpha_f

    # Blend
    out = np.empty_like(frame)
    for c in range(3):
        out[:,:,c] = (alpha_f * rgb[:,:,c] + alpha_b * frame[:,:,c]).astype(frame.dtype)
    return out