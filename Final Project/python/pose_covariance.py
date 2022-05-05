import numpy as np

def estimate_pose_covariance(jacobian):
    """Returns a vector of the sqrt of the diagonal elements of the covariance, based on a 1st-order approximation"""
    # scipy.least_squares returns jacobian at the solution
    sigma_r = np.ones(max(np.shape(jacobian)[0], np.shape(jacobian)[0])) # Jacobian should be 6x2n or 2nx6, dont know which. sigma_r should be 2nx2n
    sigma_p = np.linalg.inv(jacobian.T@sigma_r@jacobian)
    return sigma_p.diagonal()
