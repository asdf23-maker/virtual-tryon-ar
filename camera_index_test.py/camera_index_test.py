import cv2

for i in range(-1, 4):  # test -1, 0, 1, 2, 3
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
        cap = cv2.VideoCapture(i, backend)  # try different backends
        print(f"Index {i}, backend {backend}: opened? {cap.isOpened()}")
        cap.release()
