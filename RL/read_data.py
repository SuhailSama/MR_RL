# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:09:51 2022

@author: suhai
"""

import glob, os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from MR_env import MR_Env
from scipy import spatial

pickle_dir = "D:\Projects\MMRs\experimental_data\pickles"
os.chdir(pickle_dir)


def angle_between(pts1, pts2):
    delta = pts2 - pts1
    
    if len(delta.shape) == 2: 
        angles = np.arctan2(delta[:,0],delta[:,1])
    else:
        delta = delta/np.sqrt(delta[0]*delta[0]+delta[1]*delta[1])
        print(delta)
        angles = np.arctan2(delta[1],delta[0])
    return angles # % (2*np.pi) 


file = 'closedloop6.pickle'

dataset = pd.read_pickle(file)[0]
   

p_f = -1    
x           = np.expand_dims(np.array(dataset['PositionX']) ,axis =1)
y           = np.expand_dims(np.array(dataset['PositionY']) ,axis =1)
xy          = np.concatenate((x,y), axis=1)                              [:p_f]
targets_x   = np.expand_dims(np.array(dataset["TrajectoryX"]) ,axis =1)
targets_y   = np.expand_dims(np.array(dataset["TrajectoryY"]) ,axis =1)
target_xy   = np.concatenate((targets_x,targets_y), axis=1)             [:p_f]

Track_Pars  = dataset['Track_Params(frame,error,current_pos,target_pos)']
Coil_Output = dataset['Coil Output(frame,Bx,By,Bz,rolling_frequency,alpha, gamma)']


frames       = np.array([x[0] for x in Track_Pars]) [:p_f]
error        = np.array([x[1] for x in Track_Pars]) [:p_f]
current_pos  = np.array([x[2] for x in Track_Pars]) [:p_f]
target_pos   = np.array([x[3] for x in Track_Pars]) [:p_f]
# alpha        = angle_between(current_pos,target_pos)

### model 

T = len(x)
env = MR_Env()
episode_action_obs = []
observation = env.reset(current_pos[0,:])
error_tol = 10 # min(error)
pt = current_pos[0]
xy_model = np.array([pt])

target_idx = 1 

nearest_node = target_pos[target_idx]
distance = ((pt[0] - nearest_node[0] )**2 + (pt[1] - nearest_node[1])**2)**0.5

for t in range(T):  
    count =0 
    while (distance > error_tol) and (count <5): 
        alpha = angle_between(pt, nearest_node)
        action = np.array([5, alpha])
        # print("cur_pos : ",pt.round(decimals=1),", dist : ",round(distance,2) , 
        #       ", alpha : ", round(np.rad2deg(alpha)), ", Moving to node # ",
        #       nearest_node.round(decimals=1))
        observation, reward, done = env.step(action)
        pt = observation[:2]
        xy_model = np.append(xy_model,np.array([pt]) , axis=0)
        distance = ((pt[0] - nearest_node[0] )**2 + (pt[1] - nearest_node[1])**2)**0.5
        count += 1
    target_idx +=  1 
    if target_idx >= target_pos.shape[0]-1: 
        break 
    nearest_node = target_pos[target_idx]
    distance = ((pt[0] - nearest_node[0] )**2 + (pt[1] - nearest_node[1])**2)**0.5
    # print("REACHED!! cur_pos : ",pt.round(decimals=1),", dist : ",round(distance,2) , 
              # ", alpha : ", np.rad2deg(alpha).round(decimals=1), ", next node ",
              # nearest_node.round(decimals=1))

    # print ("Actions: rolling_frequesncy: %2.2f, alpha : %5.2f" % (action[0],action[1]))
 

### plot stuff 
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)

ax.plot(xy[:,0],   xy[:,1],                 'g-' ,linewidth=5.0, label='MR xy')
ax.plot(xy[0,0], xy[0,1],                   'go',linewidth=1.0, label='start MR_xy')
ax.plot(target_xy[:,0], target_xy[:,1],       'bx',linewidth=2.0, label='reference xy')
# ax.plot(current_pos[:,0], current_pos[:,1], 'b--',linewidth=1.0, label='xy_experiment')
# ax.plot(target_pos[:,0], target_pos[:,1],   'b.',linewidth=.01, label='target_pos')
ax.plot(xy_model[:,0], xy_model[:,1],   'r--',linewidth=1, label='xy_model')
ax.plot(xy_model[0,0], xy_model[0,1],   'ro',linewidth=1, label='start xy_model')
# Set the x and y axis to display a fixed range
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1000])
ax.legend()

# fig2, ax2 = plt.subplots(1, 1)
# ax2.plot(alpha)


