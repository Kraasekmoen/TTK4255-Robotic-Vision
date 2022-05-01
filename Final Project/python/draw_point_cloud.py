import matplotlib.pyplot as plt
import numpy as np

def draw_frame(ax, T, scale):
    X0 = T@np.array((0,0,0,1))
    X1 = T@np.array((1,0,0,1))
    X2 = T@np.array((0,1,0,1))
    X3 = T@np.array((0,0,1,1))
    ax.plot([X0[0], X1[0]], [X0[2], X1[2]], [X0[1], X1[1]], color='#FF7F0E')
    ax.plot([X0[0], X2[0]], [X0[2], X2[2]], [X0[1], X2[1]], color='#2CA02C')
    ax.plot([X0[0], X3[0]], [X0[2], X3[2]], [X0[1], X3[1]], color='#1F77B4')

def draw_point_cloud(X, T_m2q, xlim, ylim, zlim, colors, marker_size, frame_size):
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    if colors.max() > 1.1:
        colors = colors.copy()/255
    ax.scatter(X[0,:], X[2,:], X[1,:], c=colors, marker='.', s=marker_size, depthshade=False)
    draw_frame(ax, np.linalg.inv(T_m2q), scale=frame_size)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.title('[Click, hold and drag with the mouse to rotate the view]')
