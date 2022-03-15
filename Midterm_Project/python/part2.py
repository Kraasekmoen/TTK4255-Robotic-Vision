import numpy as np
import matplotlib.pyplot as plt
from common import *
from platform_ext import Platform
from scipy.optimize import least_squares

# Tip: The solution from HW4 is inside common.py


K = np.loadtxt('../data/K.txt')
u = np.loadtxt('../data/platform_corners_image.txt')
X = np.loadtxt('../data/platform_corners_metric.txt')
I = plt.imread('../data/video0000.jpg') # Only used for plotting

n = u.shape[1]
uv1 = np.vstack((u, np.ones(n)))
xy = np.linalg.inv(K) @ uv1
H = estimate_H(xy[:2,:], X[:2,:])
# 2.1b--------------------------------------------------------------------
T1 , T2 = decompose_H(H)

if np.sum(T1[2,:]) >= np.sum(T2[2,:]):
    T_hat = T1
else:
    T_hat = T2
u_hat = project(K, T_hat@X)
# ------------------------------------------------------------------------
# Example: Compute predicted image locations and reprojection errors
#T_hat = translate(-0.3, 0.1, 1.0)@rotate_x(1.8)

# 2.1a--------------------------------------------------------------------
# X1 = np.ones([3,4])
# X1[:2,:] = X[:2,:]
# u_hat = project(K, H@X1)
# ------------------------------------------------------------------------
errors = np.linalg.norm(u - u_hat, axis=0)

#R = T_hat[:3,:3]
#t = T_hat[:3,3]
P = K @ T_hat[:3,:]
p = np.reshape(P,[-1,1])
platform = Platform()
# r = np.hstack([u_hat[0,:] -u[0,:], u_hat[1,:] - u[1,:]])
p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
all_r = []
all_p = []
for i in range(len(X)):
    resfun = lambda p : platform.residual_fun( p[0], p[1], p[2], p[3], p[4], p[5])

    p = least_squares(resfun, x0=p, method='lm').x

    all_r.append(resfun(p))
    all_p.append(p)

all_p = np.array(all_p)
all_r = np.array(all_r)

T_hat, u_hat = platform.new_rotation()

split_r = np.array_split(all_r[-1,:],2)
new_r = np.vstack([split_r[0], split_r[1]])
errors = np.linalg.norm(new_r, axis=0)
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
#draw_frame(K, T_hat, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
plt.xlim([200, 500])
plt.ylim([600, 350])

# Tip: To see the entire image:
# plt.xlim([0, I.shape[1]])
# plt.ylim([I.shape[0], 0])

# Tip: To save the figure:
plt.savefig('out_part2_2.png')

plt.show()

