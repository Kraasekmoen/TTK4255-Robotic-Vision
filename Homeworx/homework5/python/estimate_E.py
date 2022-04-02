import numpy as np

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    Q = np.ones((n, 9))
    x1 = xy1[0,:]
    x2 = xy2[0,:]
    y1 = xy1[1,:]
    y2 = xy2[1,:]

    print("Number of point corr.:",n)

    # Some implementation of the 8-point algorithm
    """
    The Epipolar constraint is defined by x~2*E*x~1 = 0.  Assuming the input vectors xy1,xy2 contains 
    n point correspondances, this expression can be rearranged into a nx9 matrix Q times 9D vector e, which 
    is a flattened E. Because of homogeny, this will result in 8 linear equations, which can be solved
    to find / estimate E.   Q*e=0

    A minimal of 8 correspondances are needed for a solution to exist. If there are more than 8, some 
    slightly more complex methid is needed.
    """

    if n<8:
        print("Estimate_E error: Not enough point correspondances;",n,"- Returned eye.")
        return np.eye(3) # Placeholder, replace with your implementation

    # Construct Q matrix:
    Q[:,:8] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1]).T
    u, _, vh = np.linalg.svd(Q)
    _, s, _ = np.linalg.svd(Q.T@Q)

    return vh[-1,:].reshape(3,3)
    
    if n==8:
        return 1

    if n>8:
        return 1
