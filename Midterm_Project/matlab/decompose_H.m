% Solution from Homework 4
function [T1,T2] = decompose_H(H)
    H = H / norm(H(:,1));
    r1 = H(:,1);
    r2 = H(:,2);
    r3 = cross(r1, r2);
    t = H(:,3);
    R1 = closest_rotation_matrix([r1 r2 r3]);
    R2 = closest_rotation_matrix([-r1 -r2 r3]);
    T1 = [R1 t ; 0 0 0 1];
    T2 = [R2 -t ; 0 0 0 1];
end
