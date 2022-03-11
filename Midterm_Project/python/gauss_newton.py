import numpy as np

def jacobian2point(resfun, p, epsilon):
    r = resfun(p)
    J = np.empty((len(r), len(p)))
    for j in range(len(p)):
        pj0 = p[j]
        p[j] = pj0 + epsilon
        rpos = resfun(p)
        p[j] = pj0 - epsilon
        rneg = resfun(p)
        p[j] = pj0
        J[:,j] = rpos - rneg
    return J/(2.0*epsilon)

def gauss_newton(resfun, jacfun, p0, step_size, num_steps):
    r = resfun(p0)
    J = jacfun(p0)
    p = p0.copy()
    for iteration in range(num_steps):
        A = J.T@J
        b = -J.T@r
        d = np.linalg.solve(A, b)
        p += step_size*d
        r = resfun(p)
        J = jacfun(p)
    return p
