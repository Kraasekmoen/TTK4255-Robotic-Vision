function uv = project(K, X)
    % Computes the ideal pinhole projection of a 3xN array of 3D points X 
    % using the camera intrinsic matrix K. Returns the dehomogenized
    % pixel coordinates as an array of size 2xN.
    
    %
    % Placeholder code (replace with your implementation)
    %
    N = size(X,2);
    uv = zeros(2,N);
end
