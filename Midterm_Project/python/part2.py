import numpy as np
import matplotlib.pyplot as plt
from common import *

# Tip: The solution from HW4 is inside common.py

K = np.loadtxt('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\TTK4255-Robotic-Vision\\Midterm_Project\\data\\K.txt')
u = np.loadtxt('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\TTK4255-Robotic-Vision\\Midterm_Project\\data\\platform_corners_image.txt')
X = np.loadtxt('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\TTK4255-Robotic-Vision\\Midterm_Project\\data\\platform_corners_metric.txt')
I = plt.imread('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\TTK4255-Robotic-Vision\\Midterm_Project\\quanser_image_data\\video0000.jpg') # Only used for plotting

# Example: Compute predicted image locations and reprojection errors
T_hat = translate(-0.3, 0.1, 1.0)@rotate_x(1.8)
u_hat = project(K, T_hat@X)
errors = np.linalg.norm(u - u_hat, axis=0)

# Print the reprojection errors requested in Task 2.1 and 2.2.
print('Reprojection error: ')
print('all:', ' '.join(['%.03f' % e for e in errors]))
print('mean: %.03f px' % np.mean(errors))
print('median: %.03f px' % np.median(errors))

plt.imshow(I)
plt.scatter(u[0,:], u[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
plt.scatter(u_hat[0,:], u_hat[1,:], marker='.', color='red', label='Predicted')
plt.legend()

# Tip: Draw lines connecting the points for easier understanding
plt.plot(u_hat[0,:], u_hat[1,:], linestyle='--', color='white')

# Tip: To draw a transformation's axes (only requested in Task 2.3)
draw_frame(K, T_hat, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
plt.xlim([200, 500])
plt.ylim([600, 350])

# Tip: To see the entire image:
# plt.xlim([0, I.shape[1]])
# plt.ylim([I.shape[0], 0])

# Tip: To save the figure:
# plt.savefig('out_part2.png')

plt.show()
