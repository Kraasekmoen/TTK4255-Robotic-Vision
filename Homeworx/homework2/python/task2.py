from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from common import *
import cv2

# This bit of code is from HW1.
edge_threshold = 0.015
blur_sigma     = 1
filename       = '../data/grid.jpg'
I_rgb          = plt.imread(filename)
I_rgb          = im2double(I_rgb) #Ensures that the image is in floating-point with pixel values in [0,1].
I_gray         = rgb_to_gray(I_rgb)
Ix, Iy, Im     = derivative_of_gaussian(I_gray, sigma=blur_sigma) # See HW1 Task 3.6
x,y,theta      = extract_edges(Ix, Iy, Im, edge_threshold)


# You can adjust these for better results
line_threshold = 0.18
N_rho          = 400
N_theta        = 400

###########################################
#
# Task 2.1: Determine appropriate ranges
#
###########################################
# Tip: theta is computed using np.arctan2. Check that the
# range of values returned by arctan2 matches your chosen
# ranges (check np.info(np.arctan2) or the internet docs).

print(I_rgb.shape)

rho_max   = int(np.sqrt(np.square(I_rgb.shape[0]) + np.square(I_rgb.shape[1])))
rho_min   = -rho_max
theta_min = -pi
theta_max = +pi
print(rho_max)
###########################################
#
# Task 2.2: Compute the accumulator array
#
###########################################
# Zero-initialize an array to hold our votes
H = np.zeros((N_rho, N_theta))

# 1) Compute rho for each edge (x,y,theta)
rho = x*np.cos(theta) + y*np.sin(theta) # Proman's 'no for-loop' approach

# 2) Convert to discrete row,column coordinates
row_inds = np.floor(N_rho*(rho-rho_min)/(rho_max-rho_min)).astype(int)
col_inds = np.floor(N_theta*(theta-theta_min)/(theta_max-theta_min)).astype(int)
print(row_inds.shape, max(row_inds), min(row_inds))

# 3) Increment H[row,column]
for i in range(0,len(row_inds)):
    H[row_inds[i], col_inds[i]] += 1

###########################################
#
# Task 2.3: Extract local maxima
#
###########################################
# 1) Call extract_local_maxima

# 2) Convert (row, column) back to (rho, theta)
cell_rows, cell_cols = extract_local_maxima(H, line_threshold)
maxima_rho = rho_min + cell_rows*(rho_max-rho_min)/N_rho
maxima_theta = theta_min + cell_cols*(theta_max-theta_min)/N_theta

###########################################
#
# Figure 2.2: Display the accumulator array and local maxima
#
###########################################
plt.figure()
plt.imshow(H, extent=[theta_min, theta_max, rho_max, rho_min], aspect='auto')
plt.colorbar(label='Votes')
plt.scatter(maxima_theta, maxima_rho, marker='.', color='red')
plt.title('Accumulator array')
plt.xlabel('$\\theta$ (radians)')
plt.ylabel('$\\rho$ (pixels)')
plt.savefig('out_array.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

###########################################
#
# Figure 2.3: Draw the lines back onto the input image
#
###########################################
plt.figure()
plt.imshow(I_rgb)
plt.xlim([0, I_rgb.shape[1]])
plt.ylim([I_rgb.shape[0], 0])
for theta,rho in zip(maxima_theta,maxima_rho):
    draw_line(theta, rho, color='yellow')
plt.title('Dominant lines')
plt.savefig('out_lines.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

plt.show()
