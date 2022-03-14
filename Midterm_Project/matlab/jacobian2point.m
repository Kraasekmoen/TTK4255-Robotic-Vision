function J = jacobian2point(resfun, p, epsilon)
    r = resfun(p);
    J = zeros(length(r), length(p));
    for j=1:length(p)
        pj0 = p(j);
        p(j) = pj0 + epsilon;
        rpos = resfun(p);
        p(j) = pj0 - epsilon;
        rneg = resfun(p);
        J(:, j) = rpos - rneg;
        p(j) = pj0;
    end
    J = J/(2*epsilon);
end
