import numpy as np

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    l2 = F@uv1
    l1 = F.T@uv2
    e = np.sum(uv2*l2, axis=0)
    norm1 = np.linalg.norm(l1[:2,:], axis=0)
    norm2 = np.linalg.norm(l2[:2,:], axis=0)
    return 0.5*e*(1/norm1 + 1/norm2)
