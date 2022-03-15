# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser_ext import Quanser_Ext
from plot_all import *

all_detections = np.loadtxt('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\TTK4255-Robotic-Vision\\Midterm_Project\\data\\detections.txt')
#all_detections = all_detections[:150,:]    ## Adjust number to reduce number of images used 
all_weights = all_detections[:, ::3]
num_weights = np.zeros(all_weights.shape[0])
print(num_weights.shape)
for j in range(all_weights.shape[0]):
    num_weights[j] = np.sum(all_weights[j,:])
quanser = Quanser_Ext()

K       = 26                     # Number of kinetic parameters
num_s   = 3                         # Number of state parameters
num_n   = len(all_detections)           # Number of images

l = np.array([0.1145/2, 0.325, -0.050, 0.65, -0.030])           # Initial length estimates = the ones given in task 1
l = np.zeros(K)
s = np.zeros(num_s*num_n, dtype='float')                    # Initial state estimates for 'copter angles
p = np.append(l,s)                                      # Initial p-vector

all_r = []
all_p = []

# Sparsity pattern
sparse = np.zeros((14*num_n, K+num_s*num_n))
sparse[:, :K] = 1
sparse[:, K:] = np.kron(np.eye(num_n), np.ones((14,3)))
print("Sparsity matrix: ", sparse.shape)

## Batch run
resfun = lambda p : quanser.batch_residuals_ModA(all_detections=all_detections, p=p, K=K)
p = least_squares(resfun, x0=p, jac_sparsity=sparse, verbose=2, max_nfev=30).x

# Plotting
all_p = np.array(p[K:].reshape((num_n,-1)))
all_r = np.array(resfun(p).reshape((num_n,-1)))
plot_all(all_p, all_r, all_detections, subtract_initial_offset=True, num_weights=num_weights)

plt.savefig('out_part3_Mod-A.png')
plt.show()

"""
Answer to Task 1.8:
a)
Because of the way the _residue_ function has been implemented, with the weights for valid entries, the residue of unobserved markers are set to zero. 
Followingly, the corresponding index of the jacobian matrix J will be zero too - leading to a potentially singular matrix, which prevents solving for a unique solution.
The Levenberg-Marquardt method has circumvented this problem by fixing the step length to 1 and adding a term _\mu*I_ to the normal equation. This guarantees that the
equation remains solvable and that the estimation algorithm can move past the points with insufficient observational data.

b) 
The added term \mu*I adds \mu*1 to the diagonal elements of J^T*J, with \mu having a variable magnitude. The Levenberg-Marquardt method is set up in such a way that
the \mu term decreases when approaching the minimum, and increases otherwise. This means that the J^T*J term will dominate when close to the minimum, leveraging the 
benefits of Newton-Gauss, while having the '+1' term dominate further from the minimum - where Newton-Gauss struggles.
The downside of this method, is the fixed step length of 1. While Levenberg-Marquardt is robust in ensuring that each step of the optimization remains solvable,
it suffers from a sort of 'maximum accuracy' - with no guarantees beyond 'minimum + 1'. This must be taken into account when determining the stopping criterion.
"""
