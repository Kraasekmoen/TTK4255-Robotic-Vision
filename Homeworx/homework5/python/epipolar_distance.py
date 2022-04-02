import numpy as np

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    e = np.zeros(n) # Placeholder, replace with your implementation
    return e
