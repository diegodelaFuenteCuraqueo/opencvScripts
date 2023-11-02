# The KLT tracker is an optical flow-based tracking algorithm that tracks keypoints in consecutive frames.

import cv2
import numpy as np

# Load a video file
input_source = 'chickens.mp4'  # Replace with your file path

# Create a VideoCapture object for video input
cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Convert the frame to grayscale for tracking
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initialize the KLT tracker
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

# Create a mask to visualize the tracked points
mask = np.zeros_like(frame)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using the KLT tracker
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray, gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)  # Ensure they are integers
        img = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
        img = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)

    # Display the frame with the tracks
    cv2.imshow('KLT Tracker', img)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()


