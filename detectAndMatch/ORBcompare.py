# ORB (Oriented FAST and Rotated BRIEF) algorithm

import cv2
import numpy as np

# Load the two images you want to match
image1 = cv2.imread('dogs.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dogs2.jpg', cv2.IMREAD_GRAYSCALE)

# Create an ORB object
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2)

# Find the keypoints and descriptors in both images
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Create a BFMatcher (Brute-Force Matcher) object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the first 10 matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], outImg=None)

# Display the matched image
cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

