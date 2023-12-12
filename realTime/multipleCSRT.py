import cv2

# Load a video file
input_source = 'videos/traffic.mp4'  # Replace with your file path

# Create a VideoCapture object for video input
cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect moving objects
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # Adjust the area threshold as needed
            # Draw a bounding box around the detected object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Moving Object Detection', frame)

    # Check for a key event to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

