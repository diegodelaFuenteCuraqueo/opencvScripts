import cv2
import numpy as np

# Initialize variables for object tracking
bbox = None
tracking = False
p0 = None

# Create a VideoCapture object for video input
cap = cv2.VideoCapture('chickens.mp4')  # Replace with your video file path
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Callback function for mouse click event
def set_bbox(event, x, y, flags, param):
    global bbox, tracking, p0

    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = (x, y, 0, 0)
        tracking = False
        p0 = None

# Create a window for displaying the video
cv2.namedWindow('KLT Tracker')
cv2.setMouseCallback('KLT Tracker', set_bbox)

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

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    if bbox is not None:
        x, y, w, h = bbox
        if w == 0 and h == 0:
            bbox = None
            tracking = False
        else:
            p0 = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)
            tracking = True
            bbox = None

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracking and p0 is not None:
        # Calculate optical flow using the KLT tracker
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray, gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)  # Ensure they are integers
            img = cv2.line(frame, (a, b), (c, d), (0, 0, 255), 2)
            img = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # Display the frame with the tracks
        cv2.imshow('KLT Tracker', frame)

        # Print the bounding box position to the console
        x, y, w, h = bbox
        print(f"BBox Position: (x={x}, y={y}, w={w}, h={h})")

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Display the frame without tracking until a bounding box is set
        cv2.imshow('KLT Tracker', frame)

    # Handle user input to set the bounding box
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Clear the bounding box and stop tracking
        tracking = False
        bbox = None
    elif key == ord('r'):
        # Reset the bounding box to the last selected position
        bbox = (x, y, w, h)

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

