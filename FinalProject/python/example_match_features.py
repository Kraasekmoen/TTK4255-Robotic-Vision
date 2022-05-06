import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from estimate_E_ransac import *
from F_from_E import *
from figures import *

K = np.loadtxt('../data_hw5_ext/calibration/K.txt')
I1 = cv.imread('../data_hw5_ext/IMG_8210.jpg', cv.IMREAD_GRAYSCALE)
I2 = cv.imread('../data_hw5_ext/IMG_8211.jpg', cv.IMREAD_GRAYSCALE)

'''
# Initiate ORB detector
orb = cv.ORB_create(nfeatures=4000)
# find the keypoints and descriptors with ORB
kp1_orb, des1 = orb.detectAndCompute(I1,None)
kp2_orb, des2 = orb.detectAndCompute(I2,None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 50 matches.
outImg = np.zeros_like(I1)
img3 = cv.drawMatches(I1,kp1_orb,I2,kp2_orb,matches[:50],outImg= outImg ,matchesThickness= 3 ,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
'''

# NB! This script uses a very small number of features so that it runs quickly.
# You will want to pass other options to SIFT_create. See the documentation:
# https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
sift = cv.SIFT_create(nfeatures=30000)
kp1, desc1 = sift.detectAndCompute(I1, None)
kp2, desc2 = sift.detectAndCompute(I2, None)
kp1 = np.array([kp.pt for kp in kp1])
kp2 = np.array([kp.pt for kp in kp2])

# NB! You will want to experiment with different options for the ratio test and
# "unique" (cross-check).
index_pairs, match_metric = match_features(desc1, desc2, max_ratio=0.6, unique=True)
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

#uv1 = np.vstack([best_kp1.T, np.ones(best_kp1.shape[0])])
uv1 =  np.vstack([kp1[index_pairs[:,0]].T, np.ones(index_pairs.shape[0])])
#uv2 = np.vstack([best_kp2.T, np.ones(best_kp2.shape[0])])
uv2 =  np.vstack([kp2[index_pairs[:,1]].T, np.ones(index_pairs.shape[0])])
xy1 = np.linalg.inv(K)@uv1
xy2 = np.linalg.inv(K)@uv2

E = estimate_E(xy1, xy2)

#e = epipolar_distance(F_from_E(E, K), uv1, uv2)

distance_threshold = 4.0

# Automatically computed trial count
confidence = 0.99
inlier_fraction = 0.50
num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)

E, inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)
uv1 = uv1[:,inliers]
uv2 = uv2[:,inliers]
xy1 = xy1[:,inliers]
xy2 = xy2[:,inliers]



T4 = decompose_E(E)
best_num_visible = 0
for i, T in enumerate(T4):
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    P2 = T[:3,:]
    X1 = triangulate_many(xy1, xy2, P1, P2)
    X2 = T@X1
    num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
    if num_visible > best_num_visible:
        best_num_visible = num_visible
        best_T = T
        best_X1 = X1
T = best_T
X = best_X1
print('Best solution: %d/%d points visible' % (best_num_visible, xy1.shape[1]))

I1 = cv.imread('../data_hw5_ext/IMG_8210.jpg') #, cv.IMREAD_COLOR)
I1 = cv.cvtColor(I1, cv.COLOR_BGR2RGB)
I2 = cv.imread('../data_hw5_ext/IMG_8211.jpg')
I2 = cv.cvtColor(I2, cv.COLOR_BGR2RGB)
#np.random.seed(123) # Comment out to get a random selection each time
draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)
draw_point_cloud(X, I1, uv1, xlim=[-10,+10], ylim=[-5,+5], zlim=[5,20])
plt.show()

desc1 = desc1[index_pairs[:,0]]
desc2 = desc2[index_pairs[:,1]]
desc1 = desc1[inliers]
desc2 = desc2[inliers]

np.savetxt('3D_features.txt', np.hstack([X.T, desc1, desc2]) )