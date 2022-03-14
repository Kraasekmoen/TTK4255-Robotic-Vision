% Note: You must install the Optimization toolbox to use lsqnonlin.
% This can be installed by re-running the installer for the same
% version of Matlab that you are currently using. You do not need
% to reinstall all of Matlab.
%
% See: https://i.ntnu.no/wiki/-/wiki/Norsk/Matlab

clear;
clc;

% Configure lsqnonlin options (passed as argument below).
% See doc lsqnonlin for a description of all options.
options = optimoptions(@lsqnonlin);
options.Algorithm = 'levenberg-marquardt';
options.Display = 'off'; % Don't display iteration info

all_detections = load('../data/detections.txt');
quanser = Quanser;

p = [0, 0, 0];
all_r = [];
all_p = [];
for i=1:size(all_detections, 1)
    weights = all_detections(i, 1:3:end);
    uv = [all_detections(i, 2:3:end) ;
          all_detections(i, 3:3:end) ];

    % Tip: Lambda functions can be defined inside a for-loop, defining
    % a different function in each iteration. Here we pass in the current
    % image's "uv" and "weights", which get loaded at the top of the loop.
    resfun = @(p) quanser.residuals(uv, weights, p(1), p(2), p(3));

    % Tip: Use the previous image's parameter estimate as initialization
    p = lsqnonlin(resfun, p, [], [], options);

    % Collect residuals and parameter estimate for plotting later
    all_r = [all_r ; resfun(p)'];
    all_p = [all_p ; p];
end
% Tip: See comment in plot_all.py regarding the last argument.
plot_all(all_p, all_r, all_detections, false);
