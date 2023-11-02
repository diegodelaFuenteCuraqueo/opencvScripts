import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a color range for the filter (e.g., green)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create a mask for the specified color range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply the mask to the original frame to display only the filtered color
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the filtered frame
    cv2.imshow('Color Filter', filtered_frame)

    # Exit the loop by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

