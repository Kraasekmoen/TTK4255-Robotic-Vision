# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser import Quanser
from plot_all import *

all_detections = np.loadtxt('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\TTK4255-Robotic-Vision\\Midterm_Project\\data\\detections.txt')
all_weights = all_detections[:, ::3]
num_weights = np.zeros(all_weights.shape[0])
print(num_weights.shape)
for j in range(all_weights.shape[0]):
    num_weights[j] = np.sum(all_weights[j,:])
quanser = Quanser()

p = np.array([0.0, 0.0, 0.0])
all_r = []
all_p = []
for i in range(len(all_detections)):
    weights = all_detections[i, ::3]
    uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))

    # Tip: Lambda functions can be defined inside a for-loop, defining
    # a different function in each iteration. Here we pass in the current
    # image's "uv" and "weights", which get loaded at the top of the loop.
    resfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

    # Tip: Use the previous image's parameter estimate as initialization
    p = least_squares(resfun, x0=p, method='lm').x

    # Collect residuals and parameter estimate for plotting later
    all_r.append(resfun(p))
    all_p.append(p)

all_p = np.array(all_p)
all_r = np.array(all_r)
# Tip: See comment in plot_all.py regarding the last argument.
plot_all(all_p, all_r, all_detections, subtract_initial_offset=True, num_weights=num_weights)
plt.savefig('out_part1b.png')
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
