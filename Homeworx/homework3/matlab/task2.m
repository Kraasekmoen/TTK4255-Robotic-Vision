clear;

K = load('../data/task2K.txt');
X = load('../data/task2points.txt');

% Task 2.2: Implement the project function
u = project(K, X);

% You would change these to be the resolution of your image. Here we have
% no image, so we arbitrarily choose a resolution.
width = 600;
height = 400;

%
% Figure 2.2: Show pinhole projection of 3D points
%
figure; clf;
scatter(u(1,:), u(2,:)); hold on;

% The following commands are useful when the figure is meant to simulate
% a camera image. Note: these must be called after all draw commands!!!!

axis equal;       % -> Ensures that pixels are square (no squashing)
axis ij;          % -> Places the origin in the upper left
xlim([0 width]);  % -> Prevents auto-zooming on the data
ylim([0 height]); %    (Change these if you actually want a zoom!)
