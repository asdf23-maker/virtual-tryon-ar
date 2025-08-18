import cv2
import mediapipe as mp

pose = mp.solutions.pose.Pose()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Pose Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
