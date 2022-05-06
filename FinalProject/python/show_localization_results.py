#
# This script uses example localization results to show
# what the figure should look like. You need to modify
# this script to work with your data.
#

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from draw_point_cloud import *

#model = '../example_localization'
#query = '../example_localization/query/IMG_8210'
image_name1 = 'IMG_8207'
image_name2 = 'IMG_8215'
image_name3 = 'IMG_8228'

K = np.loadtxt('../data_hw5_ext/calibration/K.txt')
I1 = cv.imread('../data_hw5_ext/'+ image_name1 +'.jpg')
I1 = cv.cvtColor(I1, cv.COLOR_BGR2RGB)
I2 = cv.imread('../data_hw5_ext/' + image_name2 +'.jpg')
I2 = cv.cvtColor(I2, cv.COLOR_BGR2RGB)
I3 = cv.imread('../data_hw5_ext/' + image_name3 +'.jpg')
I3 = cv.cvtColor(I3, cv.COLOR_BGR2RGB)


# 3D points [4 x num_points].
X_descs = np.loadtxt('./3D_features.txt')
X = X_descs[:, :4].T

# Model-to-query transformation.
# If you estimated the query-to-model transformation,
# then you need to take the inverse.
T_m2q_1 = np.loadtxt('../data_hw5_ext/'+ image_name1 +'_T_m2q.txt')
T_m2q_2 = np.loadtxt('../data_hw5_ext/'+ image_name2 +'_T_m2q.txt')
T_m2q_3 = np.loadtxt('../data_hw5_ext/'+ image_name3 +'_T_m2q.txt')

T_m2q = np.array([T_m2q_1, T_m2q_2, T_m2q_3])
# If you have colors for your point cloud model...
colors = np.loadtxt('../data_hw5_ext/colors.txt') # RGB colors [num_points x 3].
# ...otherwise...
# colors = np.zeros((X.shape[1], 3))

# These control the visible volume in the 3D point cloud plot.
# You may need to adjust these if your model does not show up.
xlim = [-10,+10]
ylim = [-10,+10]
zlim = [0,+20]

frame_size = 1;
marker_size = 5

plt.figure('3D point cloud', figsize=(10,10))
draw_point_cloud(X, T_m2q, xlim, ylim, zlim, colors=colors, marker_size=marker_size, frame_size=frame_size)
plt.tight_layout()
plt.show()


