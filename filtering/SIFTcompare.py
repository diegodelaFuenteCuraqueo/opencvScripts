# SIFT (Scale-Invariant Feature Transform) algorithm,

import cv2
import numpy as np

# Load the two images you want to match
image1 = cv2.imread('dogs1.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dogs2.jpg', cv2.IMREAD_GRAYSCALE)

# Create a SIFT object
sift = cv2.SIFT_create()

# Find the keypoints and descriptors in both images
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Create a BFMatcher (Brute-Force Matcher) object
bf = cv2.BFMatcher()

# Match the descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # k=2 means find the top 2 matches

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw the first 10 matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:10], outImg=None)

# Display the matched image
cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

