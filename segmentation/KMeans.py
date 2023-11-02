# groups pixels with similar characteristics, such as color or intensity, into clusters

import cv2
import numpy as np

image = cv2.imread('dogs2.jpg')

# Convert the image to the Lab color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Reshape the image for K-Means clustering
pixels = lab_image.reshape((-1, 3)).astype(np.float32)

k = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the label data to 8-bit for visualization
labels = labels.reshape(image.shape[:2]).astype(np.uint8)

# Create masks for each cluster
mask1 = np.uint8(labels == 0)
mask2 = np.uint8(labels == 1)
mask3 = np.uint8(labels == 2)

# Apply the masks to the original image
result1 = cv2.bitwise_and(image, image, mask=mask1)
result2 = cv2.bitwise_and(image, image, mask=mask2)
result3 = cv2.bitwise_and(image, image, mask=mask3)

# Display the segmented images
cv2.imshow('Cluster 1', result1)
cv2.imshow('Cluster 2', result2)
cv2.imshow('Cluster 3', result3)

cv2.waitKey(0)
cv2.destroyAllWindows()

