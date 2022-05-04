function draw_point_cloud(X, T_m2q, colors, my_xlim, my_ylim, my_zlim, point_size, frame_size)
    if (max(colors) > 1.1)
        colors = colors/255;
    end
    scatter3(X(1,:), X(3,:), X(2,:), point_size, colors, 'filled');
    hold on;

    s = frame_size;
    pO = T_m2q\[0 0 0 1]'; % Camera Origin
    pX = T_m2q\[s 0 0 1]'; % Camera X-axis
    pY = T_m2q\[0 s 0 1]'; % Camera Y-axis
    pZ = T_m2q\[0 0 s 1]'; % Camera Z-axis
    plot3([pO(1) pX(1)], [pO(3) pX(3)], [pO(2) pX(2)], 'r', 'linewidth', 2); % Draw camera X-axis
    plot3([pO(1) pY(1)], [pO(3) pY(3)], [pO(2) pY(2)], 'g', 'linewidth', 2); % Draw camera Y-axis
    plot3([pO(1) pZ(1)], [pO(3) pZ(3)], [pO(2) pZ(2)], 'b', 'linewidth', 2); % Draw camera Z-axis

    grid on;
    box on;
    axis equal;
    axis vis3d;
    camproj perspective;
    ylim(my_zlim);
    xlim(my_xlim);
    zlim(my_ylim);
    set(gca, 'ZDir', 'reverse');
    xlabel('X');
    ylabel('Z');
    zlabel('Y');
    h = annotation('textbox', [0 0.1 0 0], 'String', '[Hover over the figure with your mouse to access the toolbar, and select the rotator tool to rotate the view.]', 'FitBoxToText', true);
end
