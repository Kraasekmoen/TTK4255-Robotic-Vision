function H = estimate_H(xy, XY)
    % Tip: Use [U,S,V] = svd(A) for the singular value decomposition.
    % The column of V corresponding to the smallest singular value
    % is the last column, as the singular values are automatically
    % ordered by decreasing magnitude.

    n = size(XY, 2);
    A = [];

    % Tip: Append a submatrix A_i to A by
    % A = [A ; A_i];

    % Tip: Print the A matrix and make sure that no row is all zeros.

    H = eye(3); % Placeholder, replace with your implementation
end
