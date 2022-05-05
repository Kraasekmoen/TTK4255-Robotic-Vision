import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matlab_inspired_interface import match_features, show_matched_features
from common import *
from scipy.optimize import least_squares
from draw_point_cloud import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from estimate_E_ransac import *
from F_from_E import *
#from figures import *
import os
import random

K = np.loadtxt('.\\data_hw5_ext\\calibration\\K.txt')
dc = np.loadtxt('.\\data_hw5_ext\\calibration\\dc.txt')

path = '.\\data_hw5_ext'
file = '\\IMG_8221.jpg' #random.choice(os.listdir(".\\data_hw5_ext\\"))

I_rand = cv.imread('.\\data_hw5_ext\\IMG_8211.jpg', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create(nfeatures=4000)
kp3, desc3 = sift.detectAndCompute(I_rand, None)
kp3 = np.array([kp.pt for kp in kp3])

X_descs = np.loadtxt('.\\3D_features.txt')

X = X_descs[:, :4]
desc1 = X_descs[:, -256:-128].astype('float32')
desc2 = X_descs[:, -128:].astype('float32')

index_pairs13, match_metric13 = match_features(desc1, desc3, max_ratio=0.6, unique=True)
index_pairs23, match_metric23 = match_features(desc2, desc3, max_ratio=0.6, unique=True)

index_pairs = np.unique(np.concatenate((index_pairs13, index_pairs23)), axis=0)
kp_match = kp3[index_pairs[:,1]]
X_match = X[index_pairs[:,0]].astype('float32')


convex, r_vec, t_vec, inliers = cv.solvePnPRansac(objectPoints= X_match[:,:3], imagePoints= kp_match, cameraMatrix= K, distCoeffs= dc)#, flags= cv.SOLVEPNP_SQPNP)#, iterationsCount=1000 )

print(convex)
print(r_vec)
print(t_vec)
print(inliers.size)
r_vec = r_vec.reshape([-1,])
t_vec = t_vec.reshape([-1,])

identity = np.identity(4)
camera_pose = identity #rotate_z(r_vec[2]) @ rotate_y(r_vec[1]) @ rotate_x(r_vec[0]) @ translate(t_vec[0], t_vec[1] ,t_vec[2])
kp_in = kp_match[inliers].reshape([inliers.size, 2])
X_in = np.squeeze(X_match[inliers]).T

def res_fun_weight(r1, r2, r3, t1, t2 ,t3):
    pose = camera_pose @ rotate_z(r3) @ rotate_y(r2) @ rotate_x(r1) @ translate(t1, t2 ,t3)
    p = pose @ X_in
    uv_in = project(K, p)
    r = np.hstack(((uv_in[0, :].T - kp_in[:, 0]) **2, (uv_in[1, :].T - kp_in[:, 1]) **2))
    return r

p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #np.array([r_vec, t_vec]).reshape([-1,])

all_r = []
all_p = []
for i in range(X_in.shape[1]):
    resfun = lambda p : res_fun_weight( p[0], p[1], p[2], p[3], p[4], p[5])

    res = least_squares(resfun, x0=p, method='lm')
    p = res.x
    jac = res.jac

    all_r.append(resfun(p))
    all_p.append(p)

all_p = np.array(all_p)
all_r = np.array(all_r)

split_r = np.array_split(all_r[-1,:],2)
new_r = np.vstack([split_r[0], split_r[1]])
errors = np.linalg.norm(new_r, axis=0)

print('Reprojection error: ')
print('all:', ' '.join(['%.03f' % e for e in errors]))
print('mean: %.03f px' % np.mean(errors))
print('median: %.03f px' % np.median(errors))
#print('Final p values of camera: ', all_p[-1,:])
#print('Final Jacobian: ', jac)

np.savetxt('jacobian.txt', jac )
#-----------------------------------------------------------------------------------------------------------------------
#uv_cloud = np.ones([X.shape[0],3])
uv_cloud= project(K, X.T)
I1 = cv.imread('.\\data_hw5_ext\\IMG_8210.jpg', cv.COLOR_BGR2RGB)
c = I1[uv_cloud[1,:].astype(np.int32), uv_cloud[0,:].astype(np.int32), :]

colors = c / 255 #np.zeros((X_in.shape[1], 3))
#colors = np.zeros((X_in.shape[1], 3))

T_p = all_p[-1, :]
T_m2q = rotate_z(T_p[2]) @ rotate_y(T_p[1]) @ rotate_x(T_p[0]) @ translate(T_p[-3], T_p[-2] ,T_p[-1])
# These control the visible volume in the 3D point cloud plot.
# You may need to adjust these if your model does not show up.
xlim = [-10,+10]
ylim = [-10,+10]
zlim = [0,+20]

frame_size = 1
marker_size = 5
"""
plt.figure('3D point cloud', figsize=(10,10))
draw_point_cloud(X.T, T_m2q, xlim, ylim, zlim, colors=colors, marker_size=marker_size, frame_size=frame_size)
plt.tight_layout()
plt.show()
"""

def estimate_pose_covariance(jacobian):
    """Returns a vector of the sqrt of the diagonal elements of the covariance, based on a 1st-order approximation"""
    # scipy.least_squares returns jacobian at the solution
    sigma_r = np.eye(max(np.shape(jacobian)[0], np.shape(jacobian)[1])) # Jacobian should be 6x2n or 2nx6, dont know which. sigma_r should be 2nx2n
    core = jacobian.T@np.linalg.inv(sigma_r)@jacobian
    sigma_p = np.linalg.inv(core)
    return np.sqrt(sigma_p.diagonal())


### NOTE! The report wants the rotation deviance reported in degrees, so you gotta multiply with 180/pi. 
### I dunno which index of the parameter vector is which, sooo ....
covars = estimate_pose_covariance(jac)
print("Pose parameter std. deviations:\nRotations:    ", covars[:3]*(180/np.pi), "\nTranslations: ",covars[3:])
