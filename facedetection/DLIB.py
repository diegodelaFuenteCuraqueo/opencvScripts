# DLIB Face Detection
# pip install dlib

import cv2
import dlib

# Load an image
image = cv2.imread('people.jpg')

# Initialize the DLIB face detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the image
faces = detector(image, 1)

# Draw rectangles around the detected faces
for rect in faces:
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Display the image with detected faces
cv2.imshow('DLIB Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

