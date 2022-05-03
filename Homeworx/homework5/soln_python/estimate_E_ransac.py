import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *

def get_num_ransac_trials(sample_size, confidence, inlier_fraction):
    return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))

def estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials):
    uv1 = K@xy1
    uv2 = K@xy2

    print('Running RANSAC with inlier threshold %g pixels and %d trials...' % (distance_threshold, num_trials), end='')
    best_num_inliers = -1
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E_i = estimate_E(xy1[:,sample], xy2[:,sample])
        d_i = epipolar_distance(F_from_E(E_i, K), uv1, uv2)
        inliers_i = np.absolute(d_i) < distance_threshold
        num_inliers_i = np.sum(inliers_i)
        if num_inliers_i > best_num_inliers:
            best_num_inliers = num_inliers_i
            E = E_i
            inliers = inliers_i
    print('Done!')
    print('Found solution with %d/%d inliers' % (np.sum(inliers), xy1.shape[1]))
    return E, inliers
