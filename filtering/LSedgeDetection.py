# Laplacian and Sobel operators for edge detection

import cv2

# Load an image
image = cv2.imread('dogs.jpeg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Laplacian edge detection
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

# Apply Sobel edge detection in the x and y directions
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

cv2.imshow('Laplacian', laplacian)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()


