# Note: You need to install Scipy to run this script. If you don't
# want to install Scipy, then you can look for a different LM
# implementation or write your own.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from quanser import Quanser
from plot_all import *

all_detections = np.loadtxt('../data/detections.txt')
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
plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
#plt.savefig('out_part1b.png')
plt.show()

