#Edge detection is used to identify edges or boundaries in an image. OpenCV provides several edge detection algorithms. Here's how to apply the Canny edge detector:

import cv2

# Load an image
image = cv2.imread('dogs.jpeg')

# Convert the image to grayscale (required for Canny)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray_image, 100, 200)  # You can adjust the threshold values

cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

