import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from estimate_E_ransac import *
from F_from_E import *

K = np.loadtxt('.\\data\\K.txt')
I1 = plt.imread('.\\data\\image1.jpg')/255.0
I2 = plt.imread('.\\data\\image2.jpg')/255.0

# Note: You need to run without RANSAC once first to estimate the "good" E
# used in the histogram below.
# ransac = False # Part 2 and 3
ransac = True # Part 4

if ransac:
    matches = np.loadtxt('.\\data\\task4matches.txt')
else:
    matches = np.loadtxt('.\\data\\matches.txt')

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])
xy1 = np.linalg.inv(K)@uv1
xy2 = np.linalg.inv(K)@uv2

if ransac:
    e = epipolar_distance(F_from_E(np.loadtxt('E.txt'), K), uv1, uv2)
    plt.figure('Histogram of epipolar distances')
    plt.hist(np.absolute(e), range=[0, 40], bins=100, cumulative=True)
    plt.title('Cumulative histogram of |epipolar distance| using good E')
    plt.xlabel('Absolute epipolar distance (pixels)')
    plt.ylabel('Occurrences')

    distance_threshold = 4.0

    # Automatically computed trial count
    confidence = 0.99
    inlier_fraction = 0.50
    num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)

    # Alternatively, hard-coded trial count
    num_trials = 2000

    E,inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)
    uv1 = uv1[:,inliers]
    uv2 = uv2[:,inliers]
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]

E = estimate_E(xy1, xy2)

if not ransac:
    np.savetxt('E.txt', E)

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

np.random.seed(123) # Comment out to get a random selection each time
draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)
draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])
plt.show()
