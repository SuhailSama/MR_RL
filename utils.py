# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:50:39 2022

@author: suhail
"""
from scipy.ndimage import uniform_filter1d
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorsys
from MR_env import MR_Env, save_frames_as_gif
import numpy as np 
import matplotlib.cm as cm

#helper function to read some existing data
def readfile(filename):
    #extract the data
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    dataset = data[0]
    params = dataset['Track_Params(frame,error,current_pos,target_pos,alpha,time)']
    freq = data[2]
    X = []
    Y = []
    alpha = []
    time = []
    for i in range(0, len(params)):
        row = params[i]
        X.append(row[2][0])
        Y.append(row[2][1])
        alpha.append(row[4])
        time.append(row[5])
    print('Finished loading pickle file\n')
    X = np.array(X)
    Y = np.array(Y)
    alpha = np.array(alpha)
    time = np.array(time)

    return X,Y,alpha,time,freq

def run_sim(actions,init_pos=None,noise_var = 1,a0 =1):
    state_prime = np.empty((0,2))
    states      = np.empty((0,2))
    env         = MR_Env()
    state       = env.reset(init = init_pos,noise_var = noise_var,a0=a0)
    # init
    # states      = np.append(states, env.last_pos, axis=0)
    # state_prime = np.append(state_prime, np.array([0,0]), axis=0)
    for action in actions:
        env.step(action)
        states      = np.append(states, np.array([env.last_pos]), axis=0)
        state_prime = np.append(state_prime, np.array([env.state_prime]), axis=0)
    X      = states[:,0]
    Y      = states[:,1]
    alpha   = actions[:,1]
    freq    = actions[:,0]
    time    = np.linspace(0, (len(X) - 1)/30.0, len(X)) # (np.arange(len(X))) / 30.0 #timestep is 1/30
    
    return X,Y,alpha,time,freq


def plot_xy(xys, legends=[""],fig_title=[""]):
    fig, ax = plt.subplots()

    for (X,Y),legend in zip(xys,legends):
        ax.plot(X,Y, label=legend)
   
    ax.legend(loc='upper left', 
                       shadow=True, fontsize='x-small')
    fig.suptitle(fig_title[0])
    plt.show()

def plot_bounded_curves(curves, bounds, legends=[""], fig_title=[""]):
    fig, ax = plt.subplots()
    for (t, lb, ub) in bounds:
        ax.fill_between(t, lb, ub)

    colors = cm.Set1(np.linspace(0, 1, len(curves)))
    for (X,Y),legend,c in zip(curves,legends,colors):
        ax.plot(X,Y, color=c, label=legend)


    ax.legend(loc='upper left', 
                shadow=True, fontsize='x-small')
    fig.suptitle(fig_title[0])
    plt.show()   


def plot_traj(xts, legends=[""],fig_title=[""]):
    fig, ax = plt.subplots()
    for (times,xs),legend in zip(xts,legends):
        ax.plot(times,xs,label=legend)
   
    ax.legend(loc='upper left', shadow=True, fontsize='x-small')
    fig.suptitle(fig_title[0])
    plt.show()


    
def plot_vel(vxys, legends=[""],fig_title=[""]):
    fig, ax = plt.subplots()
    figy, ay = plt.subplots()
    handles =[]
    N = int(3*len(vxys))
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    colors =  list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    if N==3:
        colors =['k','b','r']
    for index,((v_desired,v_error,v_stdv,vx,vy),legend) in enumerate(zip(vxys,legends)):
        time = np.arange(len(vx))
        ax.fill_between(time,   v_desired[:,0] + v_error[:,0] - 2*v_stdv[:,0], 
                                v_desired[:,0] + v_error[:,0] + 2*v_stdv[:,0])
        ax.fill_between(time,   v_desired[:,0] + v_error[:,0] - v_stdv[:,0],   
                                v_desired[:,0] + v_error[:,0] + v_stdv[:,0])
        
        ax.plot(time, vx,                            colors[3*index+0]) #data (real from sim)
        ax.plot(time, v_desired[:,0],                colors[3*index+1]) #desired (ideal=> no noise)
        ax.plot(time, v_desired[:,0] + v_error[:,0], colors[3*index+2]) #learned (predicted from )
        
        ay.fill_between(time,   v_desired[:,1] + v_error[:,1] - 2*v_stdv[:,1], 
                                v_desired[:,1] + v_error[:,1] + 2*v_stdv[:,1])
        ay.fill_between(time,   v_desired[:,1] + v_error[:,1] - v_stdv[:,1],   
                                v_desired[:,1] + v_error[:,1] + v_stdv[:,1])
        
        ay.plot(time, vy,                            colors[3*index+0]) #data (real from sim)
        ay.plot(time, v_desired[:,1],                colors[3*index+1]) #desired (ideal=> no noise)
        ay.plot(time, v_desired[:,1] + v_error[:,1], colors[3*index+2]) #learned (predicted from )
        
        #proxy artists for figures
        h1 = mpatches.Patch(color= colors[3*index+0], label='Data_'+legend)
        h2 = mpatches.Patch(color= colors[3*index+1], label='Desired_'+legend)
        h3 = mpatches.Patch(color= colors[3*index+2], label='Learned_'+legend)
        handles.append(h1)
        handles.append(h2)
        handles.append(h3)
    
    fig.legend(handles=handles), figy.legend(handles=handles)
    fig.suptitle(fig_title[0]+'_X') ,figy.suptitle(fig_title[0]+'_Y') 
    fig.legend(loc='upper left', 
                       shadow=True, fontsize='x-small')
    figy.legend(loc='upper left', 
                       shadow=True, fontsize='x-small')
    plt.show()

    
def test_gp(gp,X,Y,a0,alpha,freq,time):
    #pretend this is happening real-time in a while loop
    v_desired = np.zeros( (len(time), 2) )
    v_error = np.zeros( (len(time), 2) )
    v_stdv  = np.zeros( (len(time), 2) )
    alpha_pred  = np.zeros( (len(time), 1) )
    for i in range(0, len(time)):
        print(a0,freq,np.cos(alpha[i]),np.sin(alpha[i]),a0*freq*np.cos(alpha[i]))
        v_desired[i,0] = a0*freq*np.cos(alpha[i])
        v_desired[i,1] = a0*freq*np.sin(alpha[i])
        alpha_t,muX, muY, sigX, sigY = gp.predict(v_desired[i,:])
        v_error[i,0] = muX
        v_error[i,1] = muY
        v_stdv[i,0] = sigX
        v_stdv[i,1] = sigY
        alpha_pred[i] = alpha_t
    
    return alpha_pred,(v_desired,v_error,v_stdv,vx,vy)

def find_alpha_corrected(v_desired,gp):
    alpha_corrected,muX, muY, sigX, sigY = gp.predict(v_desired)    
    return alpha_corrected,muX, muY, sigX, sigY
