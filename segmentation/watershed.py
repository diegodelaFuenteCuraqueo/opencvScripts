import cv2
import numpy as np

image = cv2.imread('dogs2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply a morphological operation to remove small noise and fill in small holes
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

# Find sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
dist_transform = cv2.convertScaleAbs(dist_transform)  # Convert to CV_8U data type
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Find sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Create markers for watershed
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels to distinguish sure regions
markers = markers + 1

# Mark unknown region with 0
markers[unknown == 255] = 0

# Apply the watershed algorithm
cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red

cv2.imshow('Watershed Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

