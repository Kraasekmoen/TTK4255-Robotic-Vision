from cmath import cos, sin
import numpy as np
import matplotlib.pyplot as plt

#
# Tip: Define functions to create the basic 4x4 transformations
#
# def translate_x(x): Translation along X-axis
# def translate_y(y): Translation along Y-axis
# def translate_z(z): Translation along Z-axis
# def rotate_x(radians): Rotation about X-axis
# def rotate_y(radians): Rotation about Y-axis
# def rotate_z(radians): Rotation about Z-axis
#
def translate_x_y_z(x,y,z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])
#
# Note that you should use np.array, not np.matrix,
# as the latter can cause some unintuitive behavior.
#

def rotate_x(rd):
    return np.array([[1, 0, 0],
                    [0,cos(rd), -sin(rd)],
                    [0, sin(rd), cos(rd)]])

def rotate_y(rd):
    return np.array([[cos(rd), 0, sin(rd)],
                    [0, 1, 0],
                    [-sin(rd), 0, cos(rd)]])

def rotate_z(rd):
    return np.array([[cos(rd), -sin(rd), 0],
                    [sin(rd), cos(rd), 0],
                    [0, 0, 1]])

def extrinsic_rotate_zxy(g, b, a):
    return (rotate_z(g)@rotate_x(a))@rotate_y(b)

def transform_o_to_c(g,b,a, x=0,y=0,z=0, dim=4):
    R = extrinsic_rotate_zxy(g,b,a)
    t = translate_x_y_z(x,y,z)

    T = np.zeros((dim, dim))
    T[0:dim-1,0:dim-1] = R
    T[-1,-1] = 1
    T = t@T
    return T

def data_transform(X,T):
    N = X.shape[1]
    Xo = np.zeros(X.shape)
    for i in range(0,N):
        Xo[:,i] = T@X[:,i]
    return Xo


def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """

    # Tip: Use the @ operator for matrix multiplication, the *
    # operator on arrays performs element-wise multiplication!

    N = X.shape[1]
    U = np.zeros([3,N])
    for i in range(0,N):
        U[:,i] = K@(X[0:3,i])
    uv = U[0:2,:]/U[2,:]
    return uv

def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    This uses your project function, so implement it first.

    Control the length of the axes using 'scale'.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X) # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis

def Trf_comp_z(rad, translate):
    temp = np.zeros((4,4))
    temp[0:3,0:3] = rotate_z(rad)
    temp[0:3,-1] = translate
    temp[-1,-1]=1
    return temp

def Trf_comp_y(rad, translate):
    temp = np.zeros((4,4))
    temp[0:3,0:3] = rotate_y(rad)
    temp[0:3,-1] = translate
    temp[-1,-1]=1
    return temp

def Trf_comp_x(rad, translate):
    temp = np.zeros((4,4))
    temp[0:3,0:3] = rotate_x(rad)
    temp[0:3,-1] = translate
    temp[-1,-1]=1
    return temp