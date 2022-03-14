function p = gauss_newton(resfun, jacfun, p0, step_size, num_steps)
    r = resfun(p0);
    J = jacfun(p0);
    p = p0;
    for iteration=1:num_steps
        A = J'*J;
        b = -J'*r;
        d = A \ b;
        p = p + step_size*d;
        r = resfun(p);
        J = jacfun(p);
    end
end
