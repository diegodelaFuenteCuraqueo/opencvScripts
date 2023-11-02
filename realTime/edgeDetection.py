import cv2

cap = cv2.VideoCapture('chickens.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    edge_frame = cv2.Canny(frame, 100, 200)

    cv2.imshow('Edge Detection', edge_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

