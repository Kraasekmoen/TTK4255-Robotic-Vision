function uv = project(K, X)
    % Computes the pinhole projection of a 3xN or 4xN matrix X using
    % the camera intrinsic matrix K. Returns the 2xN pixel coordinates.
    assert(size(X,1) == 3 || size(X,1) == 4, 'X must be a 3xN or 4xN matrix');
    uvw = K*X(1:3,:);
    uv = [uvw(1,:)./uvw(3,:) ; uvw(2,:)./uvw(3,:)];
end
