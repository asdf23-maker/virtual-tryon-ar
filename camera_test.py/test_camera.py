import cv2

for index in range(5):
    print(f"Trying camera index {index}...")
    cap = cv2.VideoCapture(-1, cv2.CAP_DSHOW)
  # Windows DSHOW backend
  if not cap.isOpened():
    print("❌ Camera failed to open in try-on script.")
    exit()
else:
    print("✅ Camera opened in try-on script.")


    if cap.isOpened():
        print(f"✅ Camera {index} opened successfully!")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            cv2.imshow("Webcam Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        break

    else:
        print(f"❌ Camera index {index} failed.")
