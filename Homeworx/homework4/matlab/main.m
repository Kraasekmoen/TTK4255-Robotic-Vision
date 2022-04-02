clc;
clear;

K          = load('../data/K.txt');
detections = load('../data/detections.txt');
XY         = load('../data/XY.txt')';
n_total    = size(XY,2); % Total number of markers (= 24)

fig = figure(1);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.8, 0.6]);

% for image_number=0:22 % Use this to run on all images
for image_number=4:4 % Use this to run on a single image

    % Load data
    % valid : Boolean mask where valid[i] is True if marker i was detected
    %     n : Number of successfully detected markers (<= n_total)
    %    uv : Pixel coordinates of successfully detected markers
    valid = detections(image_number + 1, 1:3:end) == 1;
    uv = [detections(image_number + 1, 2:3:end) ;
          detections(image_number + 1, 3:3:end)];
    uv = uv(:, valid);
    n = size(uv, 2);

    % Tip: The 'valid' array can be used to perform logical array indexing,
    % e.g. to extract the XY values of only those markers that were detected.
    % Use this when calling estimate_H and when computing reprojection error.

    % Tip: Helper arrays with 0 and/or 1 appended can be useful if
    % you want to replace for-loops with array/matrix operations.
    % uv1 = [uv ; ones(1,n)];
    % XY1 = [XY ; ones(1,n_total)];
    % XY01 = [XY ; zeros(1,n_total) ; ones(1,n_total)];

    xy = zeros(2, n);                 % TASK: Compute calibrated image coordinates
    H = estimate_H(xy, XY(:, valid)); % TASK: Implement this function
    uv_from_H = zeros(2, n_total);    % TASK: Compute predicted pixel coordinates using H

    [T1,T2] = decompose_H(H); % TASK: Implement this function

    T = T1; % TASK: Choose solution (try both T1 and T2 for Task 3.1, but choose automatically for Task 3.2)

    % NB! generate_figure expects the predicted pixel coordinates as 'uv_from_H'.
    clf(fig);
    generate_figure(image_number, K, T, uv, uv_from_H, XY);
    waitforbuttonpress;
end
