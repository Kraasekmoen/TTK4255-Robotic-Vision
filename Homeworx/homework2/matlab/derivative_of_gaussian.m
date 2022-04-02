function [Ix, Iy, Im] = derivative_of_gaussian(I, sigma)
    % Computes the gradient in the x and y direction using the derivatives
    % of a 2-D Gaussian, as described in HW1 Task 3.6. Returns the gradient
    % images (Ix, Iy) and the magnitude Im.
    
    h = ceil(3*sigma);
    x = linspace(-h, h, 2*h + 1);
    e = exp(-x.^2/(2*sigma^2));
    g = e/sqrt(2*pi*sigma^2);
    d = -x.*e/(sigma*sigma*sigma*sqrt(2*pi));

    Ix = zeros(size(I));
    Iy = zeros(size(I));
    for row=1:size(I,1)
        Ix(row,:) = conv(I(row,:), d, 'same');
        Iy(row,:) = conv(I(row,:), g, 'same');
    end
    for col=1:size(I,2)
        Ix(:,col) = conv(Ix(:,col), g, 'same');
        Iy(:,col) = conv(Iy(:,col), d, 'same');
    end
    Im = sqrt(Ix.^2 + Iy.^2);
end
