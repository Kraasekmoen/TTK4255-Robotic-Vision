
import numpy as np
import cv2
import scipy as sc
import matplotlib.pyplot as plt

# Task 1.1
"""
The following are point operators:
c) and d)
"""

# Task 1.2
# The following two methods were borrowed from https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
def processImage(image): 
  image = cv2.imread(image) 
  image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY) 
  return image
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
"""
The provided kernel appears to increase contrast on edges, creating a dark outline. Run script and compare grid_grayed.jpg with 2DConvolved.jpg
"""
#filename = '.\data\grid.jpg'
filename = 'C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\Homeworx\\homework1\\python\\data\\grid.jpg'
img = plt.imread(filename)
procd_img = processImage(filename)
cv2.imwrite('grid_grayed.jpg',procd_img)
kernel = np.array([[-0.5,-1.0,-0.5],[-1.0,7,-1.0],[-0.5,-1.0,-0.5]])
output = convolve2D(procd_img, kernel, padding=2)
cv2.imwrite('2DConvolved.jpg',output)
plt.axis("off")
plt.imshow(img)
plt.show()

fig, axes = plt.subplots(1,3,sharey='row')
plt.set_cmap('gray')
axes[0].imshow(img)
axes[1].imshow(procd_img)
axes[2].imshow(output)
axes[0].set_title('Original')
axes[1].set_title('Grayscaled')
axes[2].set_title('Convolved with kernel')
plt.tight_layout()
plt.show()