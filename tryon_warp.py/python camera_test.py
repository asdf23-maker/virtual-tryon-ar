import cv2

for i in range(0, 3):  # try indices 0, 1, and 2
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    print(f"Index {i} opened:", cap.isOpened())
    if cap.isOpened(): cap.release()
