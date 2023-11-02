import cv2
import numpy as np

image = cv2.imread('dogs.jpeg')

# Define the region to crop (top-left and bottom-right coordinates)
x1, y1, x2, y2 = 100, 50, 300, 250
cropped_image = image[y1:y2, x1:x2]

cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
