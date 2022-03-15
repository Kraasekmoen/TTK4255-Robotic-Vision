import matplotlib.pyplot as plt
import numpy as np
from common import *

# Extened quanser for batch optimization

class Platform:
    def __init__(self):
        self.K = np.loadtxt('../data/K.txt')
        #self.heli_points = np.loadtxt('../data/heli_points.txt').T
        self.uv = np.loadtxt('../data/platform_corners_image.txt')
        self.corners = np.loadtxt('../data/platform_corners_metric.txt')
        self.platform_to_camera = np.loadtxt('../data/platform_to_camera.txt')

    def residual_fun(self, p1, p2, p3, X, Y, Z):
        # "uv" is a 2x7 vector of the pixel coordinates of the detected markers. If a particular marker was not detected, the 'uv' entry may be invalid.
        # To avoid fuckery, the variable "weights" is a 1x7 vector, where the entry indicates which entires of 'uv' are valid or not. If an entry is invalid,
        # as determined by reading corresponding index of 'weights', you should multiply the invalid entry by 0 and proceed as usual.
        # "kinetics" is a K-D vector containing the kinetics for the transformations

        # Compute the helicopter coordinate frames
        platform_pose = self.platform_to_camera @ rotate_z(p3) @ rotate_y(p2) @ rotate_x(p1) @translate(X, Y, Z)

        # Compute the predicted image location of the markers
        p = platform_pose @ self.corners
        self.T = p
       # p1 = self.arm_to_camera @ heli_points[:,:3]            # Remove "self." to use estimates instead
        #p2 = self.rotors_to_camera @ heli_points[:,3:]
        uv_hat = project(self.K, p)
        self.uv_hat = uv_hat # Save for use in draw()

        #
        # Tip: Use np.hstack to concatenate the horizontal and vertical residual components
        # into a single 1D array. Note: The plotting code will not work correctly if you use
        # a different ordering.
        ###
        r = np.hstack(((uv_hat[0,:] - self.uv[0,:]), (uv_hat[1,:] - self.uv[1,:])))
        ###
        return r

    def new_rotation(self):
       # T = self.uv_hat @ np.linalg.inv(self.corners) @ np.linalg.inv(self.K)
        return self.T, self.uv_hat

    def draw(self, uv, weights, image_number):
        I = plt.imread('../data/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*uv[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])