import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('C:\\Users\\sindr\\Documents\\UniversiTales\\V22\\RobVis\\Homeworx\\homework1\\python\\data\\grass.jpg')
img_g = np.zeros(img.shape, dtype=float)

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if (float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2]) != 0):
            img_g[x,y,1] = float(img[x,y,0]/(float(img[x,y,0])+float(img[x,y,1])+float(img[x,y,2])))

rows = 2
cols = 2

thresholds = np.linspace(0.375,0.40,num=(rows*cols))
thresholds = thresholds.astype(float)
fig2, axs2 = plt.subplots(rows,cols)
plt.set_cmap('gray')

img_ths = np.zeros((10,720,1280))
k=0
for i in range(0,rows):
    for j in range(0,cols):
        img_ths[k,:,:] = img_g[:,:,1]
        img_ths[k,:,:] = img_ths[k,:,:] < thresholds[k]

        axs2[i,j].imshow(img_ths[k,:,:], aspect='auto')
        axs2[i,j].set_title(thresholds[k])
        axs2[i,j].axis('off')
        k+=1

plt.show()

