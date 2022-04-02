function G = gaussian(I, sigma)
    % Applies a 2-D Gaussian blur with standard deviation sigma to
    % a grayscale image I.

    % Generate the 1-D Gaussian filter kernel
    h = ceil(3*sigma);
    x = linspace(-h, h, 2*h + 1);
    g = exp(-x.^2/(2*sigma^2))/sqrt(2*pi*sigma^2);

    % Filter the image (using the fact that the Gaussian is separable)
    G = zeros(size(I));
    for row=1:size(I,1)
        G(row,:) = conv(I(row,:), g, 'same');
    end
    for col=1:size(I,2)
        G(:,col) = conv(G(:,col), g, 'same');
    end
end
