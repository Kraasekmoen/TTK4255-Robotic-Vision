import numpy as np

def triangulate_many(xy1, xy2, P1, P2):
    """
    Arguments
        xy: Calibrated image coordinates in image 1 and 2
            [shape 3 x n]
        P:  Projection matrix for image 1 and 2
            [shape 3 x 4]
    Returns
        X:  Dehomogenized 3D points in world frame
            [shape 4 x n]
    """
    n = xy1.shape[1]
    X = np.empty((4,n))
    for i in range(n):
        A = np.empty((4,4))
        A[0,:] = P1[0,:] - xy1[0,i]*P1[2,:]
        A[1,:] = P1[1,:] - xy1[1,i]*P1[2,:]
        A[2,:] = P2[0,:] - xy2[0,i]*P2[2,:]
        A[3,:] = P2[1,:] - xy2[1,i]*P2[2,:]
        U,s,VT = np.linalg.svd(A)
        X[:,i] = VT[3,:]/VT[3,3]
    return X
