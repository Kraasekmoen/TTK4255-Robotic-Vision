import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

def decompose_H(H):
    k = np.linalg.norm(H[:,0])
    H /= k
    r1 = H[:,0]
    r2 = H[:,1]
    r3 = np.cross(r1, r2) # note: r1 x r2 = -r1 x -r2 = r3
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

def closest_rotation_matrix(Q):
    """
    Find closest (in the Frobenius norm sense) rotation matrix to 3x3 matrix Q
    """
    U,s,VT = np.linalg.svd(Q)
    R = U@VT

    # A rotation matrix must satisfy transpose(R) R = I. This is the
    # constraint Zhang uses in Eq. (15) in his paper. We may check how
    # well this is satisfied using the Frobenius norm:
    print('\tFrobenius norm (I - QTQ) w/o corr.:', np.linalg.norm(np.eye(3) - Q.T@Q, ord='fro'))
    print('\tFrobenius norm (I - RTR) w/ corr.:', np.linalg.norm(np.eye(3) - R.T@R, ord='fro'))

    # We might also want to check that the determinant is +1, but this
    # by itself is not sufficient to have a valid rotation matrix. Note
    # that the determinant of an orthogonal matrix is always +1 or -1.
    # print('\tDeterminant(Q):', np.linalg.det(Q))
    # print('\tDeterminant(R):', np.linalg.det(R))
    return R

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def draw_frame(K, T, scale=1):
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
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis

def generate_figure(fig, image_number, K, T, uv, uv_predicted, XY):

    fig.suptitle('Image number %d' % image_number)

    #
    # Visualize reprojected markers and estimated object coordinate frame
    #
    I = plt.imread('../data/image%04d.jpg' % image_number)
    plt.subplot(121)
    plt.imshow(I)
    draw_frame(K, T, scale=4.5)
    plt.scatter(uv[0,:], uv[1,:], color='red', label='Detected')
    plt.scatter(uv_predicted[0,:], uv_predicted[1,:], marker='+', color='yellow', label='Predicted')
    plt.legend()
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.axis('off')

    #
    # Visualize scene in 3D
    #
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.plot(XY[0,:], XY[1,:], np.zeros(XY.shape[1]), '.')
    pO = np.linalg.inv(T)@np.array([0,0,0,1])
    pX = np.linalg.inv(T)@np.array([6,0,0,1])
    pY = np.linalg.inv(T)@np.array([0,6,0,1])
    pZ = np.linalg.inv(T)@np.array([0,0,6,1])
    plt.plot([pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color='blue')
    plt.plot([pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color='green')
    plt.plot([pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color='red')
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    ax.set_zlim([-25, 25])
    ax.set_xlabel('x')
    ax.set_zlabel('y')
    ax.set_ylabel('z')

    plt.tight_layout()
