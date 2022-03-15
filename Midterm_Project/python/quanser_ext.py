import matplotlib.pyplot as plt
import numpy as np
from common import *


# Extened quanser for batch optimization

class Quanser_Ext:
    def __init__(self):
        self.K = np.loadtxt(
            '../data/K.txt')
        self.heli_points = np.loadtxt(
            '../data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt(
            '../data/platform_to_camera.txt')

    def residuals_ModA(self, uv, weights, yaw, pitch, roll, kinetics=26):
        # "uv" is a 2x7 vector of the pixel coordinates of the detected markers. If a particular marker was not detected, the 'uv' entry may be invalid.
        # To avoid fuckery, the variable "weights" is a 1x7 vector, where the entry indicates which entires of 'uv' are valid or not. If an entry is invalid,
        # as determined by reading corresponding index of 'weights', you should multiply the invalid entry by 0 and proceed as usual.
        # "kinetics" is a K-D vector containing the kinetics for the transformations

        # Compute the helicopter coordinate frames
        base_to_platform = translate(kinetics[0], kinetics[0], 0.0) @ rotate_z(yaw)
        hinge_to_base = translate(0.00, 0.00, kinetics[1]) @ rotate_y(pitch)
        arm_to_hinge = translate(0.00, 0.00, kinetics[2])
        rotors_to_arm = translate(kinetics[3], 0.00, kinetics[4]) @ rotate_x(roll)
        self.base_to_camera = self.platform_to_camera @ base_to_platform
        self.hinge_to_camera = self.base_to_camera @ hinge_to_base
        self.arm_to_camera = self.hinge_to_camera @ arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera @ rotors_to_arm

        # Marker coordinate estimates

        heli_points = np.zeros((4, 7))  # Initiate matrix
        heli_points[-1, :] = 1  # Add homog. 'ones'
        heli_points[:3, :] = kinetics[5:26].reshape(3, 7)

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ heli_points[:, :3]  # Remove "self." to use estimates instead
        p2 = self.rotors_to_camera @ heli_points[:, 3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat  # Save for use in draw()

        #
        # Tip: Use np.hstack to concatenate the horizontal and vertical residual components
        # into a single 1D array. Note: The plotting code will not work correctly if you use
        # a different ordering.
        ###
        r = np.hstack(((uv_hat[0, :] - uv[0, :]), (uv_hat[1, :] - uv[1, :])))
        r[0:7] = r[0:7] * weights
        r[7:14] = r[7:14] * weights
        ###
        return r

    def batch_residuals_ModA(self, all_detections, p, K):
        # "all_detections" contains the 'fasit' for all the marker locations
        r = np.zeros((len(all_detections), 14))
        states = p[K:].reshape(len(p[K:]) // 3, 3)  # Reshape states into 3xN matrix for convenience
        for i in range(len(all_detections)):  # For each frame/image
            weights = all_detections[i, ::3]  # 'Which markers are visable / which detections are valid'
            uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))  # Extract detections

            r[i, :] = self.residuals_ModA(uv, weights, states[i, 0], states[i, 1], states[i, 2], kinetics=p[0:K])
        return r.reshape(14 * len(all_detections))

    def draw(self, uv, weights, image_number):
        I = plt.imread(
            '../data/video%04d.jpg' % image_number)
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

    def residuals_ModB(self, uv, weights, yaw, pitch, roll, kinetics=33):
        k = kinetics
        # Compute the helicopter coordinate frames
        # "2 is like arm and 3 is like rotor"

        one_to_platform = rotate_x(k[0]) @ rotate_y(k[1]) @ translate(k[2], k[3], 0) @ rotate_z(yaw)
        two_to_one = rotate_x(k[4]) @ rotate_y(k[5]) @ translate(k[6], 0, k[7]) @ rotate_y(pitch)
        three_to_two = rotate_x(k[8]) @ rotate_y(k[9]) @ translate(0, k[10], k[11]) @ rotate_x(roll)

        one_to_camera = self.platform_to_camera @ one_to_platform
        two_to_camera = one_to_camera @ two_to_one
        three_to_camera = two_to_camera @ three_to_two

        # Marker coordinate estimates
        heli_points = np.zeros((4, 7))  # Initiate matrix
        heli_points[-1, :] = 1  # Add homog. 'ones'
        heli_points[:3, :] = kinetics[12:33].reshape(3, 7)

        # Compute the predicted image location of the markers
        p1 = two_to_camera @ heli_points[:, :3]  # Remove "self." to use estimates instead
        p2 = three_to_camera @ heli_points[:, 3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat  # Save for use in draw()

        r = np.hstack(((uv_hat[0, :] - uv[0, :]), (uv_hat[1, :] - uv[1, :])))
        r[0:7] = r[0:7] * weights
        r[7:14] = r[7:14] * weights
        ###
        return r

    def batch_residuals_ModB(self, all_detections, p, K):
        # "all_detections" contains the 'fasit' for all the marker locations
        r = np.zeros((len(all_detections), 14))
        states = p[K:].reshape(len(p[K:]) // 3, 3)  # Reshape states into 3xN matrix for convenience
        for i in range(len(all_detections)):  # For each frame/image
            weights = all_detections[i, ::3]  # 'Which markers are visable / which detections are valid'
            uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))  # Extract detections

            r[i, :] = self.residuals_ModB(uv, weights, states[i, 0], states[i, 1], states[i, 2], kinetics=p[0:K])
        return r.reshape(14 * len(all_detections))

    def residuals_ModC(self, uv, weights, yaw, pitch, roll, kinetics=39):
        k = kinetics
        # Compute the helicopter coordinate frames
        # "2 is like arm and 3 is like rotor"

        one_to_platform = rotate_x(k[0]) @ rotate_y(k[1]) @ rotate_z(k[2]) @ translate(k[3], k[4], k[5]) @ rotate_z(yaw)
        two_to_one = rotate_x(k[6]) @ rotate_y(k[7]) @ rotate_z(k[8]) @ translate(k[9], k[10], k[11]) @ rotate_y(pitch)
        three_to_two = rotate_x(k[12]) @ rotate_y(k[13]) @ rotate_z(k[14]) @ translate(k[15], k[16], k[17]) @ rotate_x(
            roll)

        one_to_camera = self.platform_to_camera @ one_to_platform
        two_to_camera = one_to_camera @ two_to_one
        three_to_camera = two_to_camera @ three_to_two

        # Marker coordinate estimates
        heli_points = np.zeros((4, 7))  # Initiate matrix
        heli_points[-1, :] = 1  # Add homog. 'ones'
        heli_points[:3, :] = kinetics[18:39].reshape(3, 7)

        # Compute the predicted image location of the markers
        p1 = two_to_camera @ heli_points[:, :3]  # Remove "self." to use estimates instead
        p2 = three_to_camera @ heli_points[:, 3:]
        uv_hat = project(self.K, np.hstack([p1, p2]))
        self.uv_hat = uv_hat  # Save for use in draw()

        r = np.hstack(((uv_hat[0, :] - uv[0, :]), (uv_hat[1, :] - uv[1, :])))
        r[0:7] = r[0:7] * weights
        r[7:14] = r[7:14] * weights
        ###
        return r

    def batch_residuals_ModC(self, all_detections, p, K):
        # "all_detections" contains the 'fasit' for all the marker locations
        r = np.zeros((len(all_detections), 14))
        states = p[K:].reshape(len(p[K:]) // 3, 3)  # Reshape states into 3xN matrix for convenience
        for i in range(len(all_detections)):  # For each frame/image
            weights = all_detections[i, ::3]  # 'Which markers are visable / which detections are valid'
            uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))  # Extract detections

            r[i, :] = self.residuals_ModC(uv, weights, states[i, 0], states[i, 1], states[i, 2], kinetics=p[0:K])
        return r.reshape(14 * len(all_detections))