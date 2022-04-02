clear;

% This bit of code is from HW1.
filename       = '../data/grid.jpg';
edge_threshold = 0.015;
blur_sigma     = 1;
I_rgb          = im2double(imread(filename));
I_gray         = rgb_to_gray(I_rgb);
[Ix,Iy,Im]     = derivative_of_gaussian(I_gray, blur_sigma);
[x,y,theta]    = extract_edges(Ix, Iy, Im, edge_threshold);

% You can adjust these for better results
line_threshold = 0.2;
N_rho          = 200;
N_theta        = 200;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 2.1: Determine appropriate ranges
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tip: theta is computed using atan2. Check that the range
% returned by atan2 matches your chosen ranges.
rho_max   = 500; % Placeholder value
rho_min   = -100; % Placeholder value
theta_min = -1; % Placeholder value
theta_max = +1; % Placeholder value

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 2.2: Compute the accumulator array
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zero-initialize an array to hold our votes
H = zeros(N_rho, N_theta);

% 1) Compute rho for each edge (x,y,theta)
% Tip: You can do this without for-loops

% 2) Convert to discrete row,column coordinates
% Tip: Use round(...) to round a number to an integer type
% Tip: Remember that Matlab indices start from 1, not 0

% 3) Increment H[row,column]
% Tip: Make sure that you don't try to access values at indices outside
% the valid range: [1,N_rho] and [1,N_theta]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 2.3: Extract local maxima
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Call extract_local_maxima

% 2) Convert back to continuous rho,theta quantities
maxima_rho = [100]; % Placeholder
maxima_theta = [0]; % Placeholder

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 2.2: Display the accumulator array and local maxima
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
imagesc(H, 'XData', [theta_min theta_max], 'YData', [rho_min rho_max]); hold on;
cb = colorbar();
cb.Label.String = 'Votes';
scatter(maxima_theta, maxima_rho, 100, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
xlabel('\theta (radians)');
ylabel('\rho (pixels)');
title('Accumulator array');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 2.3: Draw the lines back onto the input image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
imshow(I_rgb); hold on;
for i=1:size(maxima_rho)
    draw_line(maxima_theta(i), maxima_rho(i));
end
title('Dominant lines');
