import cv2

for i in range(4):  # checking indices 0 to 3
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # or try without DSHOW
    print(f"Index {i}: opened? {cap.isOpened()}")
    cap.release()
