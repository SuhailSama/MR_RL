import pickle

from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import LearningModule as GP # type: ignore
from MR_env import MR_Env, save_frames_as_gif

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

def run_sim(actions,init_pos=None):
    state_prime = np.empty((0,2))
    states      = np.empty((0,2))
    env         = MR_Env()
    state       = env.reset(init = init_pos)
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
    time    = np.arange(len(X))
    
    return X,Y,alpha,time,freq


def run_exp(actions):
    # TODO: Max will update
    pass 
    
    
    return X,Y,alpha,time,freq

def plot_traj(X1,Y1,X2=[],Y2=[], legends=["experiment"]):
    fig, ax = plt.subplots()

    ax.plot(X1,Y1, label=legends[0])
    if (X2!=[]) and (Y2!=[]):
        ax.plot(X2,Y2, label=legends[1])
    ax.legend(loc='upper left', 
                       shadow=True, fontsize='x-small')
    plt.show()
def test_gp(gp,X,Y,a0,alpha,freq,time,fig_title):
    #pretend this is happening real-time in a while loop
    N = (int)(1 / 0.035) #filter position data due to noisy sensing
    
    v_desired = np.zeros( (len(time), 2) )
    v_error = np.zeros( (len(time), 2) )
    v_stdv  = np.zeros( (len(time), 2) )
    
    
    for i in range(N, len(time)):
        X_hist = X[0:i]
        Y_hist = Y[0:i]
    
        # smooth out our X history
        X_hist = uniform_filter1d(X_hist, N, mode="nearest")
        Y_hist = uniform_filter1d(Y_hist, N, mode="nearest")
        #calculate velocity via position derivative
        vx = np.gradient(X_hist, time[0:i])
        vy = np.gradient(Y_hist, time[0:i])
        # apply smoothing to the velocity signal
        vx = uniform_filter1d(vx, N, mode="nearest")
        vy = uniform_filter1d(vy, N, mode="nearest")
    
        muX, muY, sigX, sigY = gp.predict(alpha[i])
    
        v_error[i,0] = muX
        v_error[i,1] = muY
        v_stdv[i,0] = sigX
        v_stdv[i,1] = sigY
    
        v_desired[i,0] = a0*freq*np.cos(alpha[i])
        v_desired[i,1] = a0*freq*np.sin(alpha[i])
    
    # now we can finally smooth out and plot the velocity + prediction
    X = uniform_filter1d(X, N, mode="nearest")
    Y = uniform_filter1d(Y, N, mode="nearest")
    # calculate velocity via position derivative
    vx = np.gradient(X, time)
    vy = np.gradient(Y, time)
    # apply smoothing to the velocity signal
    vx = uniform_filter1d(vx, N, mode="nearest")
    vy = uniform_filter1d(vy, N, mode="nearest")
    
    plt.figure()
    
    plt.fill_between(time,  v_desired[:,0] + v_error[:,0] - 2*v_stdv[:,0],  v_desired[:,0] + v_error[:,0] + 2*v_stdv[:,0])
    plt.fill_between(time,  v_desired[:,0] + v_error[:,0] - v_stdv[:,0],    v_desired[:,0] + v_error[:,0] + v_stdv[:,0])
    
    plt.plot(time, v_desired[:,0], 'b')
    plt.plot(time, v_desired[:,0] + v_error[:,0], 'r')
    plt.plot(time, vx, 'k')
    
    #proxy artists for figures
    h1 = mpatches.Patch(color='k', label='Data')
    h2 = mpatches.Patch(color='b', label='Desired')
    h3 = mpatches.Patch(color='r', label='Learned')
    
    plt.legend(handles=[h1, h2, h3])
    plt.title(fig_title)
    plt.show()    
    
### Start the code!
case = "sim"


if case == "data": 
    px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed2.pickle')
    todel = np.argwhere(alpha >= 500)
    if len(todel) > 0:
        todel = int(todel[0])
        alpha = alpha[0:todel-1]
        px = px[0:todel-1]
        py = py[0:todel-1]
        time = time[0:todel-1]
    plot_traj(px,py,legends =["experiment"])
    # create a LearningModule object
    gp = GP.LearningModule()
    #train by passing in raw position + control signals as well as the time stamp
    #note that freq is constant
    a0 = gp.learn(px, py, alpha,freq, time)
    print("Estimated a0 value is " + str(a0))
    
    #this function plots what the GP has learned for each axis
    gp.visualize()
elif case == "sim":
    time_steps =600
    actions = np.array([[1, np.pi*(2*(t/time_steps)-1)*(-1)**(t//600)] 
                        for t in range(1,time_steps)]) # [T,action_dim]
    px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions)
    plot_traj(px_sim,py_sim,legends =["experiment"])
    # create a LearningModule object
    gp_sim = GP.LearningModule()
    #train by passing in raw position + control signals as well as the time stamp
    a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim,freq_sim[0], time_sim)
    print("Estimated a0 value is " + str(a0_sim))
    gp_sim.visualize()
    
elif case == "exp" :
    # TODO: Max will update the run_exp function
    time_steps =600
    actions = np.array([[1, np.pi*(2*(t/time_steps)-1)*(-1)**(t//600)] 
                        for t in range(1,time_steps)]) # [T,action_dim]
    px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_exp(actions)
    plot_traj(px_sim,py_sim,legends =["experiment"])
    # create a LearningModule object
    gp_sim = GP.LearningModule()
    #train by passing in raw position + control signals as well as the time stamp
    a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim,freq_sim[0], time_sim)
    print("Estimated a0 value is " + str(a0_sim))
    gp_sim.visualize()
else: 
    px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed2.pickle')
    todel = np.argwhere(alpha >= 500)
    if len(todel) > 0:
        todel   = int(todel[0])
        alpha   = alpha[2:todel-1]
        px      = px[2:todel-1]
        py      = py[2:todel-1]
        time    = time[2:todel-1]
    
    freq_sim    = freq*np.ones(px.shape)/38
    actions = np.array([[a,b] for a,b in zip(freq_sim,alpha)])
    px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions,init_pos = np.array([px[0],py[0]]))

    plot_traj(px,py,
              px_sim,py_sim,
              legends =["experiment","simulation"])

    # create a LearningModule object
    gp = GP.LearningModule()
    gp_sim = GP.LearningModule()
    #train by passing in raw position + control signals as well as the time stamp
    #note that freq is constant
    a0 = gp.learn(px, py, alpha, freq, time_sim)
    a0_sim = gp_sim.learn(px_sim,py_sim,alpha_sim,freq_sim[0],time_sim)
    print("Estimated a0 value is " + str(a0))
    print("Estimated a0_sim value is " + str(a0_sim))
    
    #this function plots what the GP has learned for each axis
    gp.visualize()
    gp_sim.visualize()
######################### TESTING ########################
##########################################################
# read some more data and see how well we can predict
if case == "data": 
    px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed.pickle')
    todel = np.argwhere(alpha >= 500)
    if len(todel) > 0:
        todel   = int(todel[0])
        alpha   = alpha[2:todel-1]
        px      = px[2:todel-1]
        py      = py[2:todel-1]
        time    = time[2:todel-1]
    plot_traj(px,py,legends =["experiment"])
    time = time - time[0] #start at t=0 for readability of the final graph
    fig_title = ["experiment"]
    test_gp(gp,px,py,a0,alpha,freq,time,fig_title[0])

elif case == "sim":
    time_steps = 40
    time_sim
    actions = np.array([[2, 0.1*np.pi*t**2] for t in range(1,time_steps)]) # [T,action_dim]
    px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions)
    plot_traj(px_sim,py_sim,
              legends =["simulation"])
    
    time = np.arange(1,time_steps) #start at t=0 for readability of the final graph
    fig_title = ["simulation"]
    alpha_sim = actions[:,1]
    freq_sim = actions[0,0]
    test_gp(gp_sim,px_sim,py_sim,a0_sim,alpha_sim,freq_sim[0],time_sim,fig_title[0])
elif case == "exp" :
    time_steps = 40
    time_sim
    actions = np.array([[2, 0.1*np.pi*t**2] for t in range(1,time_steps)]) # [T,action_dim]
    px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_exp(actions)
    plot_traj(px_sim,py_sim,
              legends =["simulation"])
    
    time = np.arange(1,time_steps) #start at t=0 for readability of the final graph
    fig_title = ["simulation"]
    alpha_sim = actions[:,1]
    freq_sim = actions[0,0]
    test_gp(gp_sim,px_sim,py_sim,a0_sim,alpha_sim,freq_sim[0],time_sim,fig_title[0])
    
else: 
    px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed.pickle')
    todel = np.argwhere(alpha >= 500)
    if len(todel) > 0:
        todel   = int(todel[0])
        alpha   = alpha[2:todel-1]
        px      = px[2:todel-1]
        py      = py[2:todel-1]
        time    = time[2:todel-1]
    freq_sim    = freq*np.ones(px.shape)/38
    actions = np.array([[a,b] for a,b in zip(freq_sim,alpha)])
    px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions,init_pos = np.array([px[0],py[0]]))
    
    plot_traj(px,py,
              px_sim,py_sim,
              legends =["experiment","simulation"])
    
    time = time_sim #time - time[0] #start at t=0 for readability of the final graph
    fig_title = ["experiment","simulation"]
    test_gp(gp,px,py,a0,alpha,freq,time,fig_title[0])
    test_gp(gp_sim,px_sim,py_sim,a0_sim,alpha_sim,
            freq_sim[0],time_sim,fig_title[1])


