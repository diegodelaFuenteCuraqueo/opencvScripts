import cv2

image = cv2.imread('dogs.jpeg')

# Resize the image to a new width and height
new_width, new_height = 300, 200
resized_image = cv2.resize(image, (new_width, new_height))

cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

