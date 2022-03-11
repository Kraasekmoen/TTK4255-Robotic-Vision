import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('.\\data\\K.txt')
        self.heli_points = np.loadtxt('.\\data\\heli_points.txt').T
        self.platform_to_camera = np.loadtxt('.\\data\\platform_to_camera.txt')

    def residuals(self, uv, weights, yaw, pitch, roll):
        # "uv" is a 2x7 vector of the pixel coordinates of the detected markers. If a particular marker was not detected, the 'uv' entry may be invalid.
        # To avoid fuckery, the variable "weights" is a 1x7 vector, where the entry indicates which entires of 'uv' are valid or not. If an entry is invalid,
        # as determined by reading corresponding index of 'weights', you should multiply the invalid entry by 0 and proceed as usual.

        # Compute the helicopter coordinate frames
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat # Save for use in draw()

        #
        # TASK: Compute the vector of residuals.
        #
        # Tip: Use np.hstack to concatenate the horizontal and vertical residual components
        # into a single 1D array. Note: The plotting code will not work correctly if you use
        # a different ordering.
        r = np.zeros(2*self.heli_points.shape[1]) # Placeholder, remove me!
        #for i in range(0, self.heli_points.shape[1]):
        print(self.heli_points.shape, self.uv_hat.shape, r.shape)

        return r

    def draw(self, uv, weights, image_number):
        I = plt.imread('.\\data\\video%04d.jpg' % image_number)
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
