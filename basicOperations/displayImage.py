import cv2

# Load an image
image = cv2.imread('images/dogs.jpeg')

# Display the image
cv2.imshow('Loaded Image', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

