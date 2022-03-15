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

K       = 33                     # Number of kinetic parameters
num_s   = 3                         # Number of state parameters
num_n   = len(all_detections)           # Number of images

l = np.zeros(K)                                     # Initial length estimates = the ones given in task 1
s = np.zeros(num_s*num_n, dtype='float')        # Initial state estimates for 'copter angles
p = np.append(l,s)                          # Initial p-vector

all_r = []
all_p = []

# Sparsity pattern
sparse = np.zeros((14*num_n, K+num_s*num_n))
sparse[:, :K] = 1
sparse[:, K:] = np.kron(np.eye(num_n), np.ones((14,3)))
print("Sparsity matrix: ", sparse.shape)

## Batch run
resfun = lambda p : quanser.batch_residuals_ModB(all_detections=all_detections, p=p, K=K)
p = least_squares(resfun, x0=p, jac_sparsity=sparse, verbose=2, max_nfev=150).x

# Plotting
all_p = np.array(p[K:].reshape((num_n,-1)))
all_r = np.array(resfun(p).reshape((num_n,-1)))
plot_all(all_p, all_r, all_detections, subtract_initial_offset=True, num_weights=num_weights)

plt.savefig('out_part3_Mod-B.png')
plt.show()
