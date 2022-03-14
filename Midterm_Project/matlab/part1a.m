clear;
clc;

image_number = 40; % Image to run on (must be in the range [0, 350])
p0 = [0, 0, 0];    % Initial parameters (yaw, pitch, roll)
step_size = 0.9;   % Gauss-Newton step size
num_steps = 10;    % Gauss-Newton iterations
epsilon = 1e-6;    % Finite-difference epsilon

% Task 1.3:
% Comment out these two lines after testing your implementation
% of the "residuals" method.
%
image_number = 0;
p0 = [11.6, 28.9, 0.0]*pi/180;

% Tip:
% Here, "uv" is a 2x7 array of detected marker locations.
% It is the same size in every image, but some of its
% entries may be invalid if the corresponding markers were
% not detected. Which entries are valid is encoded in
% the "weights" array, which is a 1D array of length 7.
%
n = 7; % Number of markers
detections = load('../data/detections.txt');
weights = detections(image_number + 1, 1:3:end);
uv = [detections(image_number + 1, 2:3:end) ;
      detections(image_number + 1, 3:3:end) ];

quanser = Quanser;

% Tip:
% Many optimization packages for Matlab expect you to provide a
% callable function that computes the residuals, and optionally
% the Jacobian, at a given parameter vector. The provided Gauss-Newton
% implementation also follows this practice. However, because the
% "residuals" method takes arguments other than the parameters, you
% must first define a "lambda function wrapper" that takes only a
% single argument (the parameter vector), and likewise for computing
% the Jacobian. This can be done as follows. Note that the Jacobian
% uses the 2-point finite differences method.
%
resfun = @(p) quanser.residuals(uv, weights, p(1), p(2), p(3));
jacfun = @(p) jacobian2point(resfun, p, epsilon);

% You must use a different image to run the rest of the script
if image_number == 0
    disp('Residuals at image 0:');
    resfun(p0)
    return
end

p = gauss_newton(resfun, jacfun, p0, step_size, num_steps);

% Calculate and print the reprojection errors at the optimum
r = resfun(p);
r2d = [r(1:n)' ; r(n+1:2*n)']; % Convert back to 2xn matrix...
e = vecnorm(r2d); % ... so that we can compute Euclidean lengths
fprintf('Reprojection errors at solution:\n')
for i=1:length(e)
    fprintf('Marker %d: %5.02f px\n', i, e(i));
end
fprintf('Average:  %5.02f px\n', mean(e));
fprintf('Median:   %5.02f px\n', median(e));

% Visualize the frames and marker points
quanser.draw(uv, weights, image_number);
