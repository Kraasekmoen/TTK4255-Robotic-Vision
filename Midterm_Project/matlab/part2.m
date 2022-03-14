% Tip: The solution from HW4 is inside the folder
% (estimate_H, decompose_H and closest_rotation_matrix).

clear;
clc;

K = load('../data/K.txt');
u = load('../data/platform_corners_image.txt');
X = load('../data/platform_corners_metric.txt');
I = imread('../data/video0000.jpg'); % Only used for plotting

% Example: Compute predicted image locations and reprojection errors
T_hat = translate(-0.3, 0.1, 1.0)*rotate_x(1.8);
u_hat = project(K, T_hat*X);
errors = vecnorm(u_hat - u);

% Print the reprojection errors requested in Task 2.1 and 2.2.
fprintf('Reprojection errors:\n');
fprintf('all: '); fprintf('%.3f ', errors); fprintf('px\n');
fprintf('mean: %.03f px\n', mean(errors));
fprintf('median: %.03f px\n', median(errors));

figure; clf;
imshow(I);
hold on;
scatter(u(1,:), u(2,:), 100, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white', 'LineWidth', 1.5);
scatter(u_hat(1,:), u_hat(2,:), 30, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');

% Tip: Draw lines connecting the points for easier understanding
plot([u_hat(1,:) u_hat(1,1)], [u_hat(2,:) u_hat(2,1)], 'w--', 'linewidth', 1.5);

% Tip: To draw a transformation's axes (only requested in Task 2.3)
draw_frame(K, T_hat, 0.05, true);

legend('Detected', 'Predicted');

% Tip: To zoom in on the platform:
xlim([200, 500]);
ylim([350, 600]);

% Tip: To see the entire image:
% xlim([0, size(I, 2)]);
% ylim([0, size(I, 1)]);
