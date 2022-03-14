function plot_all(all_p, all_r, detections, subtract_initial_offset)
    % Tip: The logs have been time-synchronized with the image sequence,
    % but there may be an offset between the motor angles and the vision
    % estimates. You may optionally subtract that offset by passing true
    % to subtract_initial_offset.

    %
    % Print reprojection error statistics
    %
    reprojection_errors = [];
    for i=1:size(all_p,1)
        weights = detections(i, 1:3:end);
        r = reshape(all_r(i,:), [], 2)';
        e = vecnorm(r);
        e = e(weights==1); % Keep only valid reprojection errors
        reprojection_errors = [reprojection_errors e];
    end
    fprintf('Reprojection error over whole image sequence:\n');
    fprintf('- Maximum: %.04f pixels\n', max(reprojection_errors));
    fprintf('- Average: %.04f pixels\n', mean(reprojection_errors));
    fprintf('- Median: %.04f pixels\n', median(reprojection_errors));

    %
    % Figure: Reprojection error distribution
    %
    fig = figure(2);
    clf(fig);
    histogram(reprojection_errors, 'NumBins', 80);
    ylabel('Frequency');
    xlabel('Reprojection error (pixels)');
    title('Reprojection error distribution');

    %
    % Figure: Comparison between logged encoder values and vision estimates
    %
    logs      = load('../data/logs.txt');
    enc_time  = logs(:,1);
    enc_yaw   = logs(:,2);
    enc_pitch = logs(:,3);
    enc_roll  = logs(:,4);

    vis_yaw = all_p(:,1);
    vis_pitch = all_p(:,2);
    vis_roll = all_p(:,3);
    if subtract_initial_offset
        vis_yaw = vis_yaw - (vis_yaw(1) - enc_yaw(1));
        vis_pitch = vis_pitch - (vis_pitch(1) - enc_pitch(1));
        vis_roll = vis_roll - (vis_roll(1) - enc_roll(1));
    end

    vis_fps  = 16;
    enc_frame = enc_time*vis_fps;
    vis_frame = 0:(size(all_p,1)-1);

    fig = figure(3);
    clf(fig);

    subplot(311);
    plot(enc_frame, enc_yaw); hold on;
    plot(vis_frame, vis_yaw);
    legend('Encoder log', 'Vision estimate');
    xlim([0, vis_frame(end)]);
    ylim([-1, 1]);
    ylabel('Yaw (radians)');
    title('Helicopter trajectory');

    subplot(312);
    plot(enc_frame, enc_pitch); hold on;
    plot(vis_frame, vis_pitch);
    xlim([0, vis_frame(end)]);
    ylim([-0.25, 0.6]);
    ylabel('Pitch (radians)');

    subplot(313);
    plot(enc_frame, enc_roll); hold on;
    plot(vis_frame, vis_roll);
    xlim([0, vis_frame(end)]);
    ylim([-0.6, 0.6]);
    ylabel('Roll (radians)');
    xlabel('Image number');
end
