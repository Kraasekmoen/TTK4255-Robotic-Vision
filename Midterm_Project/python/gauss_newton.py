import numpy as np
import copy

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

def gauss_newton(resfun, jacfun, p0, step_size, num_steps, xtol):
    print("Running GN for ", num_steps, " steps ...")
    r = resfun(p0)
    J = jacfun(p0)
    p = p0.copy()
    diff = 99999
    for iteration in range(num_steps):
        A = J.T@J
        b = -J.T@r
        d = np.linalg.solve(A, b)
        p_new =p + step_size*d
        diff = np.linalg.norm(p_new - p)
        print("A: ", A)
        if diff<xtol:
            print("GN: Stopped early at iteration ",iteration+1," of ", num_steps, ". Diff: ", diff)
            return p
        r = resfun(p_new)
        J = jacfun(p_new)
        p = p_new
    print("GN: Completed all steps. Smallest diff: ", diff)
    return p
