clear;
clc;

% Note: You must save the calibration session in the app by
% clicking the "Save Session" button. Note that this is not
% the same as "Export Camera Parameters". You may save the
% session to the calibration folder. Otherwise, you need to
% change the path below.

session = load('../data_hw5_ext/calibration/calibrationSession.mat').calibrationSession;

params = session.CameraParameters;
detections = session.BoardSet.BoardPoints;
image_size = session.BoardSet.ImageSize; % height,width

displayErrors(session.EstimationErrors, params);

subplot(121);
for i=1:size(params.ReprojectionErrors, 3)
    scalar_errors = vecnorm(params.ReprojectionErrors(:,:,i), 2, 2);
    mean_error = mean(scalar_errors);
    mean_errors(i) = mean_error;
end
bar(mean_errors);
title('Mean reprojection error');
xlabel('Image number');
ylabel('Reprojection error (pixels)');

subplot(122);
for i=1:size(detections, 3)
    u = detections(:,1,i);
    v = detections(:,2,i);
    hold on;
    scatter(u, v);
end
axis equal;
axis ij;
grid on;
box on;
xlim([0, image_size(2)]);
ylim([0, image_size(1)]);
xlabel('u (pixels)');
ylabel('v (pixels)');
title('All corner detections');
