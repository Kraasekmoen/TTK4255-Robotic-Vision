import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np


# 2.1   Print the dimensions of an image file
img = plt.imread('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\Homeworx\\homework1\\python\\data\\grass.jpg')
print("Size of image is ", img.shape[0],'x',img.shape[1])

# 2.2 Isolate the RGB channels of a color image
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
img_R = np.copy(img)
img_G = np.copy(img)
img_B = np.copy(img)

img_R[:,:,1:3] = 0
img_G[:,:,0] = 0
img_G[:,:,2] = 0
img_B[:,:,0:2] = 0


fig, axs = plt.subplots(1,3)
axs[0].imshow(img_R)
axs[1].imshow(img_G)
axs[2].imshow(img_B)
#plt.show()

# 2.3 Thresholding
img = plt.imread('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\Homeworx\\homework1\\python\\data\\grass.jpg')
# Values are between 0 and 255 (8-bit color)
thresholds = np.linspace(40,220,num=10)
thresholds = thresholds.astype(int)
fig2, axs2 = plt.subplots(2,5)
plt.set_cmap('gray')

img_ths = np.zeros((10,720,1280,3))
k=0
for i in range(0,2):
    for j in range(0,5):
        img_ths[k,:,:,:] = img[:,:,:]
        img_ths[k,:,:,0] = 0
        img_ths[k,:,:,1] = img_ths[k,:,:,1] > thresholds[k]
        img_ths[k,:,:,2] = 0

        axs2[i,j].imshow(img_ths[k,:,:,:], aspect='auto')
        axs2[i,j].set_title(thresholds[k])
        axs2[i,j].axis('off')
        k+=1

#plt.show()


# Task 2.4 Normalized color coordinates
img = plt.imread('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\Homeworx\\homework1\\python\\data\\grass.jpg')

img_r = np.zeros(img.shape, dtype=float)
img_g = np.zeros(img.shape, dtype=float)
img_b = np.zeros(img.shape, dtype=float)


for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if (float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2]) != 0):
            img_r[x,y,0] = float(img[x,y,0]/(float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2])))
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if (float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2]) != 0):
            img_g[x,y,1] = float(img[x,y,0]/(float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2])))
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if (float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2]) != 0):
            img_b[x,y,2] = float(img[x,y,0]/(float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2])))

print(img_r[0:4,0:4,0])

fig3, axs3 = plt.subplots(1,3)
axs3[0].imshow(img_r)
axs3[0].axis('off')
axs3[1].imshow(img_g)
axs3[1].axis('off')
axs3[2].imshow(img_b)
axs3[2].axis('off')
plt.show()