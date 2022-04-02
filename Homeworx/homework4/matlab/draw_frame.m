function draw_frame(K, T, scale)
    % Visualize the coordinate frame axes of the 4x4 object-to-camera
    % matrix T using the 3x3 intrinsic matrix K.
    %
    % This uses your project function, so implement it first.
    %
    % Control the length of the axes by specifying the scale argument.

    uvO = project(K, T*[0 0 0 1]');
    uvX = project(K, T*[scale 0 0 1]');
    uvY = project(K, T*[0 scale 0 1]');
    uvZ = project(K, T*[0 0 scale 1]');
    plot([uvO(1) uvX(1)], [uvO(2) uvX(2)], 'color', '#cc4422', 'linewidth', 2); % X-axis
    plot([uvO(1) uvY(1)], [uvO(2) uvY(2)], 'color', '#11ff33', 'linewidth', 2); % Y-axis
    plot([uvO(1) uvZ(1)], [uvO(2) uvZ(2)], 'color', '#3366ff', 'linewidth', 2); % Z-axis
end
