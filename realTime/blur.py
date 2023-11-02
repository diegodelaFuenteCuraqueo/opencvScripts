import cv2

# Function to change blur level based on mouse position
def update_blur_level(x):
    global blur_level
    blur_level = x

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Create a window to display the webcam feed
cv2.namedWindow('Gaussian Blur')

# Create a trackbar to control the blur level
cv2.createTrackbar('Blur Level', 'Gaussian Blur', 0, 20, update_blur_level)

# Initialize the blur level
blur_level = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Ensure that the blur level is always an odd number
    kernel_size = (15 + 2 * blur_level, 15 + 2 * blur_level)

    # Apply Gaussian blur to the frame based on the updated kernel size
    blurred_frame = cv2.GaussianBlur(frame, kernel_size, 0)

    # Display the blurred frame
    cv2.imshow('Gaussian Blur', blurred_frame)

    # Exit the loop by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

