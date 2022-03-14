function draw_frame(K, T, scale, labels)
    % Visualize the coordinate frame axes of the 4x4 object-to-camera
    % matrix T using the 3x3 intrinsic matrix K.
    %
    % Control the length of the axes by specifying the scale argument.

    if ~exist('labels', 'var')
        labels = false;
    end

    uvO = project(K, T*[0 0 0 1]');
    uvX = project(K, T*[scale 0 0 1]');
    uvY = project(K, T*[0 scale 0 1]');
    uvZ = project(K, T*[0 0 scale 1]');
    plot([uvO(1) uvX(1)], [uvO(2) uvX(2)], 'color', '#cc4422', 'linewidth', 2); % X-axis
    plot([uvO(1) uvY(1)], [uvO(2) uvY(2)], 'color', '#11ff33', 'linewidth', 2); % Y-axis
    plot([uvO(1) uvZ(1)], [uvO(2) uvZ(2)], 'color', '#3366ff', 'linewidth', 2); % Z-axis
    if labels
        text(uvX(1), uvX(2), 'X', 'color', 'white', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        text(uvY(1), uvY(2), 'Y', 'color', 'white', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        text(uvZ(1), uvZ(2), 'Z', 'color', 'white', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end
