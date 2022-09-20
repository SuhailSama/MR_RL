# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:18:54 2022

@author: suhai
"""
import numpy as np

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

A = (1, 0)
B = (1, -1)

print(angle_between(A, B))
# 45.

print(angle_between(B, A))



# pickle_files = []
# data = []

# for file in glob.glob("*.pickle"):
#     pickle_files.append(file)
#     # dataset = pd.read_pickle(file)[0]
#     # print("########################" )
#     # print("data from file: ", file )
#     # print("########################" )
    
#     # KEYS = dataset.keys()
#     # print(KEYS)
#     # for key in KEYS:
#     #     key_vals = dataset[key]
#     #     print("########################" )
#     #     print(key)
#     #     print( "\n type: ", type(key_vals))
#     #     # print( "\n shape: ", key_vals)
        

# print(pickle_files)



# def animate(i):
#     ax.clear()
#     # Get the point from the points list at index i
#     point = points[i]
#     # Plot that point using the x and y coordinates
#     ax.plot(point[0], point[1], 'go')
#     ax.plot(target_pos[:,0], target_pos[:,1], 'ro')
#     # Set the x and y axis to display a fixed range
#     ax.set_xlim([0, 1000])
#     ax.set_ylim([0, 1000])
# ani = FuncAnimation(fig, animate, frames=len(points),
#                     interval=500, repeat=False)
# plt.close()


# ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
