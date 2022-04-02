function [row,col] = extract_local_maxima(H, threshold)
    % Returns the row and column of cells whose value is strictly greater than its
    % 8 immediate neighbors, and greater than or equal to a threshold. The threshold
    % is specified as a fraction of the maximum array value.
    %
    % Note: This returns (row,column) coordinates.
    
    absolute_threshold = threshold*max(H,[],'all');
    M = imregionalmax(H) & (H >= absolute_threshold);
    [row,col,~] = find(M);
end