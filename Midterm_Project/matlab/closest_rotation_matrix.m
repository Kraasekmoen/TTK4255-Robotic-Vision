% Solution from Homework 4
function R = closest_rotation_matrix(Q)
    [U,S,V] = svd(Q);
    R = U*V';
end
