function generate_figure(image_number, K, T, uv, uv_from_H, XY)
    %
    % Visualize reprojected markers and estimated object coordinate frame
    %
    subplot(121);
    I = imread(sprintf('../data/image%04d.jpg', image_number));
    imshow(I);
    hold on;
    scatter(uv(1,:), uv(2,:), 60, 'red', 'filled');
    scatter(uv_from_H(1,:), uv_from_H(2,:), 60, 'yellow', '+', 'LineWidth', 2);
    draw_frame(K, T, 7);
    legend('Detected', 'Predicted');
    title(sprintf('Image number %d', image_number));
    xlabel('[Press any button to continue to the next image. Close the figure to exit the loop.]');

    %
    % Visualize scene in 3D
    %
    subplot(122);
    plot3(XY(1,:), XY(2,:), zeros(1,size(XY,2)), 'o');
    hold on;
    pO = T\[0 0 0 1]'; % Compute camera origin
    pX = T\[6 0 0 1]'; % Compute camera X-axis
    pY = T\[0 6 0 1]'; % Compute camera Y-axis
    pZ = T\[0 0 6 1]'; % Compute camera Z-axis
    plot3([pO(1) pX(1)], [pO(2) pX(2)], [pO(3) pX(3)], 'r', 'linewidth', 2); % Draw camera X-axis
    plot3([pO(1) pY(1)], [pO(2) pY(2)], [pO(3) pY(3)], 'g', 'linewidth', 2); % Draw camera Y-axis
    plot3([pO(1) pZ(1)], [pO(2) pZ(2)], [pO(3) pZ(3)], 'b', 'linewidth', 2); % Draw camera Z-axis
    view(30,30);
    xlim([-30 30]);
    ylim([-30 30]);
    zlim([-30 30]);
    axis('vis3d');
    camproj('perspective');
    grid('on');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
end
