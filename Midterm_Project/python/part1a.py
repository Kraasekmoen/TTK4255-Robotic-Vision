import matplotlib.pyplot as plt
import numpy as np
from gauss_newton import jacobian2point, gauss_newton
from quanser import Quanser

image_number = 40              # Image to run on (must be in the range [0, 350])
p0 = np.array([0.0, 0.0, 0.0]) # Initial parameters (yaw, pitch, roll)
step_size = 0.9                # Gauss-Newton step size
num_steps = 10                 # Gauss-Newton iterations
epsilon = 1e-6                 # Finite-difference epsilon

# Task 1.3:
# Comment out these two lines after testing your implementation
# of the "residuals" method.
#
image_number = 0
p0 = np.array([11.6, 28.9, 0.0])*np.pi/180

# Tip:
# Here, "uv" is a 2x7 array of detected marker locations.
# It is the same size in every image, but some of its
# entries may be invalid if the corresponding markers were
# not detected. Which entries are valid is encoded in
# the "weights" array, which is a 1D array of length 7.
#
detections = np.loadtxt('.\\data\\detections.txt')
weights = detections[image_number, ::3]
uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))

quanser = Quanser()

# Tip:
# Many optimization libraries for Python expect you to provide a
# callable function that computes the residuals, and optionally
# the Jacobian, at a given parameter vector. The provided Gauss-Newton
# implementation also follows this practice. However, because the
# "residuals" method takes arguments other than the parameters, you
# must first define a "lambda function wrapper" that takes only a
# single argument (the parameter vector), and likewise for computing
# the Jacobian. This can be done as follows. Note that the Jacobian
# uses the 2-point finite differences method, defined in gauss_newton.py.
#
resfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])
jacfun = lambda p : jacobian2point(resfun, p, epsilon=epsilon)

# You must use a different image to run the rest of the script
if image_number == 0:
    print('Residuals at image 0:')
    print(resfun(p0))
    quit()

p = gauss_newton(resfun=resfun, jacfun=jacfun, p0=p0, step_size=step_size, num_steps=num_steps)

# Calculate and print the reprojection errors at the optimum
r = resfun(p).reshape((2,-1))
e = np.linalg.norm(r, axis=0)
print('Reprojection errors at solution:')
for i,e_i in enumerate(e):
    print('Marker %d: %5.02f px' % (i + 1, e_i))
print('Average:  %5.02f px' % np.mean(e))
print('Median:   %5.02f px' % np.median(e))

# Visualize the frames and marker points
quanser.draw(uv, weights, image_number)
plt.savefig('out_part1a.png')
plt.show()
