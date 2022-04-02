import matplotlib.pyplot as plt
import numpy as np
from common import *

K           = np.loadtxt('../data/K.txt')
detections  = np.loadtxt('../data/detections.txt')
XY          = np.loadtxt('../data/XY.txt').T
n_total     = XY.shape[1]

fig = plt.figure(figsize=plt.figaspect(0.35))

# for image_number in range(23): # Use this to run on all images
for image_number in [4]: # Use this to run on a single image

    print('Image number %d' % image_number)

    valid = detections[image_number, 0::3] == True
    uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
    uv = uv[:,valid]
    n = uv.shape[1]

    uv1 = np.vstack((uv, np.ones(n)))
    XY1 = np.vstack((XY, np.ones(n_total)))
    XY01 = np.vstack((XY, np.zeros(n_total), np.ones(n_total)))

    xy = np.linalg.inv(K)@uv1
    xy = xy[:2,:]/xy[2,:]
    H = estimate_H(xy, XY[:,valid])
    uv_from_H = (K@H@XY1)
    uv_from_H = uv_from_H[:2,:]/uv_from_H[2,:]

    e = np.linalg.norm(uv - uv_from_H[:,valid], axis=0)
    print('\tReprojection error: %.2f (mean), %.2f (min), %.2f (max) [pixels]' % (np.mean(e), np.min(e), np.max(e)))

    T1,T2 = decompose_H(H)

    Z1 = (T1@XY01)[2,valid]
    Z2 = (T2@XY01)[2,valid]
    if np.all(Z1 > 0):
        print('\tChoosing T1')
        T = T1
    else:
        print('\tChoosing T2')
        T = T2

    plt.clf()
    generate_figure(fig, image_number, K, T, uv, uv_from_H, XY)
    plt.savefig('../data/out%04d.png' % image_number)
