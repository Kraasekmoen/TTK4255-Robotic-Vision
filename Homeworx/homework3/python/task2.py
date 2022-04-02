from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from common import *

# Tip: Use np.loadtxt to load data into an array
K = np.loadtxt('.\\data\\task2K.txt')
X = np.loadtxt('.\\data\\task3points.txt')

g = 0              # z
b = pi/4           # y
a = pi/12          # x
tx = 0
ty= 0
tz = 6

T = transform_o_to_c(g,b,a,tx,ty,tz)
X = data_transform(X,T)

# Task 2.2: Implement the project function
u,v = project(K, X)

# You would change these to be the resolution of your image. Here we have
# no image, so we arbitrarily choose a resolution.
width,height = 600,400

#
# Figure for Task 2.2: Show pinhole projection of 3D points
#
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!!!!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.title('Task 3.2')
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
draw_frame(K,transform_o_to_c(g,b,a,0,0,6))
plt.show()
