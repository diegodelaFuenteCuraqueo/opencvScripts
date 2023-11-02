import cv2

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

# Initialize variables for tracking
bbox = None
tracking = False

# Create a CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Define a function to select the initial region to track
def select_object():
    global bbox, tracking
    bbox = cv2.selectROI('Object Tracking', frame)
    tracking = True
    # Initialize the tracker with the selected region
    tracker.init(frame, bbox)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # If tracking is enabled, update the tracker with the current frame
    if tracking:
        ret, bbox = tracker.update(frame)

    # If the bounding box exists, draw it on the frame
    if bbox:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Tracking', frame)

    # Check for a key event to start tracking or exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not tracking:
        select_object()
    elif key == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

