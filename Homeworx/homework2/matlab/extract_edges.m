function [x, y, theta] = extract_edges(Ix, Iy, Im, threshold)
    % Returns the x and y coordinates of pixels whose gradient
    % magnitude is greater than the threshold.

    [y,x] = find(Im > threshold);
    index = sub2ind(size(Im), y, x);
    theta = atan2(Iy(index), Ix(index));
end
