% Note: the sample image is naturally grayscale
clear;
I = rgb_to_gray(im2double(imread('../data/calibration.jpg')));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 3.1: Compute the Harris-Stephens measure
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma_D = 1;
sigma_I = 3;
alpha = 0.06;
response = zeros(size(I)); % Placeholder

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 3.4: Extract local maxima
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
corners_y = [0]; % Placeholder
corners_x = [0]; % Placeholder

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 3.1: Display Harris-Stephens corner strength
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
imshow(response, []); % Specifying [] makes Matlab auto-scale the intensity
cb = colorbar();
cb.Label.String = 'Corner strength';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 3.4: Display extracted corners
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
imshow(I); hold on;
scatter(corners_x, corners_y, 15, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'yellow');
