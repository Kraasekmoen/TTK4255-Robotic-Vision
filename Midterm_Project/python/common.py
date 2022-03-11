import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    X = np.reshape(X, [X.shape[0],-1]) # Needed to support N=1
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def draw_frame(K, T, scale=1, labels=False):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color='#cc4422') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='#11ff33') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='#3366ff') # Z-axis
    if labels:
        textargs = {'color': 'w', 'va': 'center', 'ha': 'center', 'fontsize': 'x-small', 'path_effects': [PathEffects.withStroke(linewidth=1.5, foreground='k')]}
        plt.text(u[1], v[1], 'x', **textargs)
        plt.text(u[2], v[2], 'y', **textargs)
        plt.text(u[3], v[3], 'z', **textargs)

# Solution from Homework 4
def estimate_H(xy, XY):
    n = XY.shape[1]
    A = []
    for i in range(n):
        X,Y = XY[:,i]
        x,y = xy[:,i]
        A.append(np.array([X,Y,1, 0,0,0, -X*x, -Y*x, -x]))
        A.append(np.array([0,0,0, X,Y,1, -X*y, -Y*y, -y]))
    A = np.array(A)
    U,s,VT = np.linalg.svd(A)
    h = VT[8,:]
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], h[8]]
    ])
    # Alternatively:
    # H = np.reshape(h, [3,3])
    return H

# Solution from Homework 4
def decompose_H(H):
    k = np.linalg.norm(H[:,0])
    H /= k
    r1 = H[:,0]
    r2 = H[:,1]
    r3 = np.cross(r1, r2)
    t  = H[:,2]
    R1 = closest_rotation_matrix(np.array([r1, r2, r3]).T)
    R2 = closest_rotation_matrix(np.array([-r1, -r2, r3]).T)
    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[:3,:3] = R1
    T1[:3,3] = t
    T2[:3,:3] = R2
    T2[:3,3] = -t
    return T1, T2

# Solution from Homework 4
def closest_rotation_matrix(Q):
    U,s,VT = np.linalg.svd(Q)
    R = U@VT
    return R
