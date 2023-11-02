#  find and extract the contours (boundaries) of objects in an image

import cv2

image = cv2.imread('cars.jpg', 0)  # Load image in grayscale
_, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 2)  # Draw all detected contours
cv2.imshow('Contours', image)
cv2.waitKey(0)

