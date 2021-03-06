import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *

K = np.loadtxt('.\\data\\K.txt')
I1 = plt.imread('.\\data\\image1.jpg')/255.0
I2 = plt.imread('.\\data\\image2.jpg')/255.0
matches = np.loadtxt('.\\data\\matches.txt')
# matches = np.loadtxt('..\\data\\task4matches.txt') # Part 4

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

# Task 2: Estimate E
x1 = np.linalg.inv(K)@uv1
x2 = np.linalg.inv(K)@uv2

E = estimate_E(x1[:2,:], x2[:2,:])

# Task 3: Triangulate 3D points
# X = ...

#
# Uncomment in Task 2
#
np.random.seed(123) # Leave as commented out to get a random selection each time
draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)

#
# Uncomment in Task 3
#
# draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])

plt.show()
