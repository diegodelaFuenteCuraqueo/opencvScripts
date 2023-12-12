# OpenCV's DNN (Deep Neural Network) module

import cv2
import numpy as np

# Load an image
image = cv2.imread('images/dogs.jpeg')

# Load the pre-trained Caffe model for face detection
prototxt_path = 'facedetection/deploy.prototxt'
model_path = 'facedetection/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Resize the image to 300x300 (required for the model)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104, 117, 123))

# Pass the blob through the network to detect faces
net.setInput(blob)
detections = net.forward()

# Loop through the detected faces and draw rectangles
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Adjust the confidence threshold as needed
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        startX, startY, endX, endY = box.astype(int)  # Define these variables

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('DNN Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
