import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features

I1 = cv.imread('../data_hw5_ext/IMG_8210.jpg', cv.IMREAD_GRAYSCALE)
I2 = cv.imread('../data_hw5_ext/IMG_8211.jpg', cv.IMREAD_GRAYSCALE)

# NB! This script uses a very small number of features so that it runs quickly.
# You will want to pass other options to SIFT_create. See the documentation:
# https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
sift = cv.SIFT_create(nfeatures=4000)
kp1, desc1 = sift.detectAndCompute(I1, None)
kp2, desc2 = sift.detectAndCompute(I2, None)
kp1 = np.array([kp.pt for kp in kp1])
kp2 = np.array([kp.pt for kp in kp2])

# NB! You will want to experiment with different options for the ratio test and
# "unique" (cross-check).
index_pairs, match_metric = match_features(desc1, desc2, max_ratio=0.9, unique=False)
print(index_pairs[:10])
print('Found %d matches' % index_pairs.shape[0])

# Plot the 50 best matches in two ways
best_index_pairs = index_pairs[np.argsort(match_metric)[:50]]
best_kp1 = kp1[best_index_pairs[:,0]]
best_kp2 = kp2[best_index_pairs[:,1]]
plt.figure()
show_matched_features(I1, I2, best_kp1, best_kp2, method='falsecolor')
plt.figure()
show_matched_features(I1, I2, best_kp1, best_kp2, method='montage')
plt.show()
