# Blurring is a common image processing technique used to reduce noise or detail in an image.

import cv2

# Load an image
image = cv2.imread('dogs.jpeg')

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # You can adjust the kernel size (5, 5) and sigma (0)

cv2.imshow('Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

