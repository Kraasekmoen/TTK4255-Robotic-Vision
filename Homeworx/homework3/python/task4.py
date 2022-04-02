from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from common import *

# Task 4.2
platform_width = 0.1145
## Screw coordinates in platform frame
S0 = np.array([0,0,0])
S1 = np.array([platform_width,0,0])
S2 = np.array([platform_width,platform_width,0])
S3 = np.array([0,platform_width,0])

Screw_coords = (np.array([S0, S1, S2, S3])).T
homog_Screw_coords = np.ones((Screw_coords.shape[0]+1, Screw_coords.shape[1]))
homog_Screw_coords[0:3,:] = Screw_coords
print(homog_Screw_coords, Screw_coords.shape)

T = np.loadtxt('.\\data\\platform_to_camera.txt')
K = np.loadtxt('.\\data\\heli_K.txt')

print("T = \n",T)

## Screw coordinates in camera frame
homog_SCc = np.zeros(homog_Screw_coords.shape)          # Homogenous Screw Coordinates, camera frame - Memory allocation
for i in range(0,homog_Screw_coords.shape[1]):
    homog_SCc[:,i] = T@homog_Screw_coords[:,i]

u,v = project(K,homog_SCc)

print(homog_SCc, homog_SCc.shape, homog_SCc[:,0])

quanser = plt.imread('.\\data\\quanser.jpg')

pre_47 = True
if(pre_47):
    plt.figure()
    plt.imshow(quanser)
    plt.scatter(u, v, c='magenta',edgecolors='cyan', marker='o')
    plt.title("Task 4.2")
    plt.xlim([150, 500])
    plt.ylim([600, 400]) # The reversed order flips the figure such that the y-axis points down

    plt.show()
else:
    # Load points; 3 first in arm frame, 4 last in rotor frame
    points_raw = np.loadtxt('.\\data\\heli_points.txt')
    arm_points = (points_raw[0:3,:]).T
    rotor_points = (points_raw[3:points_raw.shape[0],:]).T

    # Transformations
    ## Translations
    tpb = np.array([0.1145/2, 0.1145/2, 0])
    tbh = np.array([0,0,0.325])
    tha = np.array([0,0,-0.05])
    tar = np.array([0.65,0,-0.03])
    ## Angles
    psi_yaw     = 11.6*(pi/180)
    theta_pith  = 28.9*(pi/180)
    phi_roll    = 0*(pi/180)
    #Transform matrices
    Tpb = Trf_comp_z(psi_yaw,tpb)
    Tbh = Trf_comp_y(theta_pith,tbh)
    Tha = Trf_comp_x(0,tha)
    Tar = Trf_comp_x(phi_roll,tar)    

    ## Arm coordinates in camera frame
    homog_arm_c = np.zeros(arm_points.shape)
    Tca = ((T@Tpb)@Tbh)@Tha
    for i in range(0,homog_arm_c.shape[1]):
        homog_arm_c[:,i] = Tca@arm_points[:,i]
    u,v = project(K,homog_arm_c)

    ## Rotor coordinates in camera frame
    Tcr = Tca@Tar
    homog_rotor_c = np.zeros(rotor_points.shape)
    for i in range(0,homog_rotor_c.shape[1]):
        homog_rotor_c[:,i] = Tcr@rotor_points[:,i]
    u2,v2 = project(K,homog_rotor_c)

    plt.figure()
    plt.imshow(quanser)
    plt.scatter(u, v, c='magenta',edgecolors='cyan', marker='o')
    plt.scatter(u2, v2, c='yellow',edgecolors='cyan', marker='o')

    plt.title("Task 4.7")

    plt.show()

