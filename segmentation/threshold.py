# egment objects from the background based on pixel intensity

import cv2

image = cv2.imread('dogs2.jpg', 0)  # Load image in grayscale
_, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)

