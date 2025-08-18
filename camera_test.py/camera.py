import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera opened:", cap.isOpened())
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
