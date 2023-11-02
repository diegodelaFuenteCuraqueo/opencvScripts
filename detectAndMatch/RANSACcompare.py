# RANSAC (Random Sample Consensus) algorithm for robust model estimation. 
# RANSAC is particularly useful when dealing with situations where there may be outliers in the feature matches

import cv2
import numpy as np

# Load the two images
image1 = cv2.imread('dogs.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dogs2.jpg', cv2.IMREAD_GRAYSCALE)

# Create a feature detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Create a Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Convert keypoints to numpy arrays
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the homography matrix using RANSAC
homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 10.0)

# Apply the mask to retain inliers
good_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]

# Visualize the inlier matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, outImg=None)
cv2.imshow('Robust Matching Result', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

