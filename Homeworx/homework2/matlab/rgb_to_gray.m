function gray = rgb_to_gray(rgb)
    % Converts a HxWx3 RGB image to a HxW grayscale image.
    gray = (rgb(:,:,1) + rgb(:,:,2) + rgb(:,:,3))/3;
end
