#
# This script uses example localization results to show
# what the figure should look like. You need to modify
# this script to work with your data.
#

import matplotlib.pyplot as plt
import numpy as np
from draw_point_cloud import *

model = '../example_localization'
query = '../example_localization/query/IMG_8210'

# 3D points [4 x num_points].
X = np.loadtxt(f'{model}/X.txt')

# Model-to-query transformation.
# If you estimated the query-to-model transformation,
# then you need to take the inverse.
T_m2q = np.loadtxt(f'{query}_T_m2q.txt')

# If you have colors for your point cloud model...
colors = np.loadtxt(f'{model}/c.txt') # RGB colors [num_points x 3].
# ...otherwise...
# colors = np.zeros((X.shape[1], 3))

# These control the visible volume in the 3D point cloud plot.
# You may need to adjust these if your model does not show up.
xlim = [-10,+10]
ylim = [-10,+10]
zlim = [0,+20]

frame_size = 1;
marker_size = 5

plt.figure('3D point cloud', figsize=(6,6))
draw_point_cloud(X, T_m2q, xlim, ylim, zlim, colors=colors, marker_size=marker_size, frame_size=frame_size)
plt.tight_layout()
plt.show()
