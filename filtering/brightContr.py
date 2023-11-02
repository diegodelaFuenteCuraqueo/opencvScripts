# Brightness and Contrast Adjustment
import cv2

# Load an image
image = cv2.imread('dogs.jpeg')

# Adjust brightness and contrast
alpha = 1.5  # Contrast control (1.0 is no change)
beta = 30  # Brightness control (0 is no change)

adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow('Adjusted Image', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

