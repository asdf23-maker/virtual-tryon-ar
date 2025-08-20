import cv2

def overlay_cloth(frame, cloth_img):
    gray = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(cloth_img, cloth_img, mask=mask)

    return cv2.add(bg, fg)
