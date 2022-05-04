%
% This script uses example localization results to show
% what the figure should look like. You need to modify
% this script to work with your data.
%

model = '../example_localization';
query = '../example_localization/query/IMG_8210';

% 3D points [4 x num_points].
X = load(sprintf('%s/X.txt', model));

% Model-to-query transformation.
% If you estimated the query-to-model transformation,
% then you need to take the inverse.
T_m2q = load(sprintf('%s_T_m2q.txt', query));

%# If you have colors for your point cloud model...
colors = load(sprintf('%s/c.txt', model)); % RGB colors [num_points x 3].
% ...otherwise...
% colors = zeros(size(X,2), 3);

% These control the visible volume in the 3D point cloud plot.
% You may need to adjust these if your model does not show up.
my_xlim = [-10,+10];
my_ylim = [-10,+10];
my_zlim = [0,+20];

% You may want to change these too.
point_size = 5;
frame_size = 1; % Length of visualized camera axes.

figure();
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.5, 0.04, 0.5, 0.6]);
draw_point_cloud(X, T_m2q, colors, my_xlim, my_ylim, my_zlim, point_size, frame_size);
