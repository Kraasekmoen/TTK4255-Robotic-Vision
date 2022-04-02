from cmath import pi
import numpy as np

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    im_out = np.zeros(I.shape[0:2])
    for x in range(0, I.shape[0]):
        for y in range(0, I.shape[1]):
            im_out[x,y] = (I[x,y,0] + I[x,y,1] + I[x,y,2])/3
    #im_out = im_out.astype(int)
    return im_out

def central_difference(I):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """

    diff_ker = np.array([0.5, 0, -0.5])
    print(diff_ker)
    h_grad = np.zeros(I.shape)
    v_grad = np.zeros(I.shape)
    immagn = np.zeros(I.shape)
    for r in range(0,I.shape[0]):
        h_grad[r,:] = np.convolve(I[r,:], diff_ker, 'same')
    for v in range(0,I.shape[1]):
            v_grad[:,v] = np.convolve(I[:,v], diff_ker, 'same')
    for (x, y), px in np.ndenumerate(I):
        immagn[x,y] = np.sqrt(np.square(h_grad[x,y]) + np.square(v_grad[x,y]))
    print(np.amax(v_grad), np.amin(v_grad), np.amax(immagn), np.amin(immagn))
    return v_grad, h_grad, immagn

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """

    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.

    h = int(np.ceil(3*sigma))
    result = np.zeros(I.shape)
    kern = np.zeros(2*h+1)
    print(h, kern.shape)

    for i in range(-h, h):
        kern[i+h] = (1/(np.sqrt(2*pi)*sigma))*np.exp(-np.square(h)/(2*np.square(sigma)))
    
    for r in range(0,I.shape[0]):
        result[r,:] = np.convolve(I[r,:], kern, 'same')
    for c in range(0, I.shape[1]):
        result[:,c] = np.convolve(result[:,c], kern, 'same')

    return result

def extract_edges(Ix, Iy, Im, threshold):
    """
    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """
    print(Ix.shape, Iy.shape, Im.shape, threshold)
    (y,x) = (Im > threshold).nonzero()
    theta = np.zeros(y.shape[0])
    print(x.shape, y.shape)
    for i in range(0,y.shape[0]):
        #print(i, x[i], y[i])
        theta[i] = np.arctan2(Iy[y[i],x[i]], Ix[y[i],x[i]])
    return x,y,theta # Placeholder
