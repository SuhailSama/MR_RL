import pickle

from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import Learning_module as GP # type: ignore
from MR_env import MR_Env, save_frames_as_gif
import colorsys

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
    time    = np.arange(len(X))
    
    return X,Y,alpha,time,freq


def run_exp(actions):
    # TODO: Max will update
    pass 
    
    
    # return X,Y,alpha,time,freq

def plot_xy(xys, legends=[""],fig_title=[""]):
    fig, ax = plt.subplots()

    for (X,Y),legend in zip(xys,legends):
        ax.plot(X,Y, label=legend)
   
    ax.legend(loc='upper left', 
                       shadow=True, fontsize='x-small')
    fig.suptitle(fig_title[0])
    plt.show()

def plot_traj(xts, legends=[""],fig_title=[""]):
    fig, ax = plt.subplots()
    for (times,xs),legend in zip(xts,legends):
        ax.plot(times,xs, label=legend)
   
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
    # N = (int)(1 / 0.035) #filter position data due to noisy sensing
    N=2
    v_desired = np.zeros( (len(time), 2) )
    v_error = np.zeros( (len(time), 2) )
    v_stdv  = np.zeros( (len(time), 2) )
    
    for i in range(N, len(time)):
        X_hist = X[0:i]
        Y_hist = Y[0:i]
    
        # smooth out our X history
        # X_hist = uniform_filter1d(X_hist, N, mode="nearest")
        # Y_hist = uniform_filter1d(Y_hist, N, mode="nearest")
        #calculate velocity via position derivative
        vx = np.gradient(X_hist, time[0:i])
        vy = np.gradient(Y_hist, time[0:i])
        # apply smoothing to the velocity signal
        # vx = uniform_filter1d(vx, N, mode="nearest")
        # vy = uniform_filter1d(vy, N, mode="nearest")
    
        muX, muY, sigX, sigY = gp.predict(alpha[i])
    
        v_error[i,0] = muX
        v_error[i,1] = muY
        v_stdv[i,0] = sigX
        v_stdv[i,1] = sigY
        print(a0,freq,np.cos(alpha[i]),np.sin(alpha[i]),a0*freq*np.cos(alpha[i]))
        v_desired[i,0] = a0*freq*np.cos(alpha[i])
        v_desired[i,1] = a0*freq*np.sin(alpha[i])
            # now we can finally smooth out and plot the velocity + prediction
    # X = uniform_filter1d(X, N, mode="nearest")
    # Y = uniform_filter1d(Y, N, mode="nearest")
    # calculate velocity via position derivative
    vx = np.gradient(X, time)
    vy = np.gradient(Y, time)
    
    # apply smoothing to the velocity signal
    # vx = uniform_filter1d(vx, N, mode="nearest")
    # vy = uniform_filter1d(vy, N, mode="nearest")
    
    return (v_desired,v_error,v_stdv,vx,vy)

def find_alpha_corrected(v_desired, v_error):
    
    v_cmd = v_desired - v_error # (~=v_sim)  ideally this should be the same as v_real?
    alpha_corrected = np.arctan2(v_cmd[:,1],v_cmd[:,0]) # start with y (see: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html)
    return alpha_corrected,v_cmd

# # if __name__ == "__main__":
# ### Start the code!
# case = "sim"

# time_steps =600
# actions = np.array([[1, np.pi*(2*(t/time_steps)-1)*(-1)**(t//600)] 
#                     for t in range(1,time_steps)]) # [T,action_dim]
# if case == "data": 
#     px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed2.pickle')
#     todel = np.argwhere(alpha >= 500)
#     if len(todel) > 0:
#         todel = int(todel[0])
#         alpha = alpha[0:todel-1]
#         px = px[0:todel-1]
#         py = py[0:todel-1]
#         time = time[0:todel-1]
#     xys =[(px,py)]
#     legends =["data"]
    
#     # create a LearningModule object
#     gp = GP.LearningModule()
#     #train by passing in raw position + control signals as well as the time stamp
#     #note that freq is constant
#     a0 = gp.learn(px, py, alpha,freq, time)
#     print("Estimated a0 value is " + str(a0))
    
#     #this function plots what the GP has learned for each axis
#     gp.visualize()
# elif case == "sim":
#     px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions)
#     xys =[(px_sim,py_sim)]
#     legends =["simulation"]
#     gp_sim = GP.LearningModule()
#     a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim,freq_sim[0], time_sim)
#     print("Estimated a0 value is " + str(a0_sim))
#     gp_sim.visualize()
    
# elif case == "exp" :
#     px_exp,py_exp,alpha_exp,time_exp,freq_exp = run_exp(actions)
#     xys =[(px_exp,py_exp)]
#     legends =["exp"]
#     gp_exp = GP.LearningModule()
#     #train by passing in raw position + control signals as well as the time stamp
#     a0_exp = gp_exp.learn(px_exp, py_exp, alpha_exp,freq_exp[0], time_exp)
#     print("Estimated a0 value is " + str(a0_exp))
#     gp_exp.visualize()
# else: #get exp data and pass alphas to simulator
#     px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed2.pickle')
#     todel = np.argwhere(alpha >= 500)
#     if len(todel) > 0:
#         todel   = int(todel[0])
#         alpha   = alpha[2:todel-1]
#         px      = px[2:todel-1]
#         py      = py[2:todel-1]
#         time    = time[2:todel-1]
    
#     freq_sim    = freq*np.ones(px.shape)/38
#     actions = np.array([[a,b] for a,b in zip(freq_sim,alpha)])
#     px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions,init_pos = np.array([px[0],py[0]]))

#     xys = [(px,py),(px_sim,py_sim)]
#     legends =["experiment","simulation"]

#     # create a LearningModule object
#     gp = GP.LearningModule()
#     gp_sim = GP.LearningModule()
#     #train by passing in raw position + control signals as well as the time stamp
#     #note that freq is constant
#     a0 = gp.learn(px, py, alpha, freq, time_sim)
#     a0_sim = gp_sim.learn(px_sim,py_sim,alpha_sim,freq_sim[0],time_sim)
#     print("Estimated a0 value is " + str(a0))
#     print("Estimated a0_sim value is " + str(a0_sim))
    
#     #this function plots what the GP has learned for each axis
#     gp.visualize()
#     gp_sim.visualize()
    
# plot_xy(xys,legends =legends)   

# ##########################################################
# ######################### TESTING ########################
# ##########################################################

# # read some more data and see how well we can predict
# time_steps = 40
# actions = np.array([[2, 0.1*np.pi*t**2] for t in range(1,time_steps)]) # [T,action_dim]
# if case == "data": 
#     px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed.pickle')
#     todel = np.argwhere(alpha >= 500)
#     if len(todel) > 0:
#         todel   = int(todel[0])
#         alpha   = alpha[2:todel-1]
#         px      = px[2:todel-1]
#         py      = py[2:todel-1]
#         time    = time[2:todel-1]
#     xys =[(px,py)]
#     legends =["data"]
#     time = time - time[0] #start at t=0 for readability of the final graph
#     fig_title = ["experiment"]
#     vxys = [test_gp(gp,px,py,a0,alpha,freq,time,fig_title[0])]
    
# elif case == "sim":
#     px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions)
#     xys =[(px_sim,py_sim)]
#     legends =["sim"]
#     fig_title = ["simulation"]
#     vxys = [test_gp(gp_sim,px_sim,py_sim,a0_sim,alpha_sim,freq_sim[0],time_sim)]
# elif case == "exp" :
#     px_exp,py_exp,alpha_exp,time_exp,freq_exp = run_exp(actions)
#     xys =[(px_exp,py_sim)]
#     legends =["exp"]
#     fig_title = ["simulation"]
#     vxys = [test_gp(gp_exp,px_exp,py_exp,a0_exp,alpha_exp,freq_exp[0],time_exp)]
    
# else: 
#     px,py,alpha,time,freq = readfile(r'D:/Projects/MMRs/Learning_Module/closedloopdata-10-1_withTIME/closed.pickle')
#     todel = np.argwhere(alpha >= 500)
#     if len(todel) > 0:
#         todel   = int(todel[0])
#         alpha   = alpha[2:todel-1]
#         px      = px[2:todel-1]
#         py      = py[2:todel-1]
#         time    = time[2:todel-1]
#     freq_sim    = freq*np.ones(px.shape)/38
#     actions = np.array([[a,b] for a,b in zip(freq_sim,alpha)])
#     px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions,init_pos = np.array([px[0],py[0]]))
    
#     xys = [(px,py),(px_sim,py_sim)]
#     legends =["experiment","simulation"]
    
#     time = time_sim #time - time[0] #start at t=0 for readability of the final graph
#     fig_title = ["experiment","simulation"]
#     vxys1 = test_gp(gp,px,py,a0,alpha,freq,time)
#     vxys2 = test_gp(gp_sim,px_sim,py_sim,a0_sim,alpha_sim,
#             freq_sim[0],time_sim)
#     vxys = [vxys1,
#             vxys2]

# plot_xy(xys,legends =legends)   
# plot_vel(vxys,legends =legends) 
##########################################################
######################### Control ########################
##########################################################
time_steps =600
actions_learn = np.array([[1, np.pi*(2*(t/time_steps)-1)*(-1)**(t//600)] 
                        for t in range(1,time_steps)]) # [T,action_dim]

time_steps = 400
actions = np.array([[1, 0.3*np.pi*((t/time_steps)-1)*(-1)**(t//100)] 
                        for t in range(1,time_steps)]) # [T,action_dim]

noise_vars= [0.1,1.0,10]
a0_def = 1
for i in range(len(noise_vars)):
    
    # BASE_train: no noise, no learning
    px_base,py_base,alpha_base,time_base,freq_base = run_sim(actions_learn,
                                                             init_pos = np.array([0,0]),
                                                             noise_var = 0.0,a0=a0_def)
    
    # sim: noise, no learning
    px_sim,py_sim,alpha_sim, time_sim,freq_sim = run_sim(actions_learn,
                                                         init_pos = np.array([0,0]),
                                                         noise_var = noise_vars[i],a0=a0_def)
    
    # learn noise and a0
    gp_sim = GP.LearningModule()
    a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim,freq_sim[0], time_sim)
    print("Estimated a0 value is " + str(a0_sim))
    gp_sim.visualize()
    
    xys  = [(px_base,py_base),
       (px_sim,py_sim),
           ]
    legends =["base (no noise)","sim with a0"
              ]
    fig_title   = ["Learn w and w/o noise"]
    plot_xy(xys,legends =legends,fig_title =["Learning traj w & w/o noise"]) 
    
    # BASE_test: no noise, no learning
    px_base,py_base,alpha_base,time_base,freq_base = run_sim(actions,
                                                             init_pos = np.array([0,0]),
                                                             noise_var = 0.0,
                                                             a0=a0_def)
    # sim: noise, no learning
    px_sim,py_sim,alpha_sim, time_sim,freq_sim = run_sim(actions,
                                                         init_pos = np.array([0,0]),
                                                         noise_var = noise_vars[i],
                                                         a0=a0_def)
    
    
    # time        = np.arange(1,time_steps) #start at t=0 for readability of the final graph
    # alpha_sim   = actions[:,1]
    
    # Case1: noise, noise correction
    vxys_1      =test_gp(gp_sim,px_sim,py_sim,
                         a0_def,alpha_sim,freq_sim[0],time_sim) # create a LearningModule object
    v_desired1,v_error1,v_stdv,vx,vy = vxys_1
    alpha_est1,v_cmd1  = find_alpha_corrected(v_desired1,v_error1)# estimate correct alphas with default a0
    actions1    = np.array([[1,alph ] for alph in alpha_est1]) # [T,action_dim]
    px_c1,py_c1,_,_,_ = run_sim(actions1,
                                init_pos = np.array([0,0]),
                                noise_var = noise_vars[i],
                                a0=a0_def)
    
    # Case3: noise, learned a0, learned error 
    vxys_2      =test_gp(gp_sim,px_sim,py_sim,
                         a0_sim,alpha_sim,freq_sim[0],time_sim) # create a LearningModule object
    v_desired2,v_error2,v_stdv,vx,vy = vxys_2
    alpha_est2,v_cmd2 = find_alpha_corrected(v_desired2, v_error2)# estimate correct alphas with learned a0
    
    actions2 = np.array([[1,alph] for alph in alpha_est2]) # [T,action_dim]
    px_c2,py_c2,_,_,_ = run_sim(actions2,
                                init_pos = np.array([0,0]),
                                noise_var = noise_vars[i],
                                a0=a0_sim)
    
    # Plot Control results 
    xys  = [(px_base,py_base),
           (px_sim,py_sim),
           (px_c1,py_c1),
           (px_c2,py_c2)
           ]
    v_sim = np.array([[i,j] for i,j in zip(vx,vy)])
    alpha_sim_recovered,_= find_alpha_corrected(v_desired1, np.zeros(v_error1.shape))
    
    alphas = [(time_sim,alpha_sim),
              (time_sim,alpha_sim_recovered),
           # (time_sim,alpha_est1),
           (time_sim,alpha_est2)
           ]
    
    vxs = [(time_sim,v_sim[:,0]),
           (time_sim,v_cmd1[:,0]),
           (time_sim,v_cmd2[:,0])
           ]
    vys = [(time_sim,v_sim[:,1]),
           (time_sim,v_cmd1[:,1]),
           (time_sim,v_cmd2[:,1])
           ]
    v_errs = [(time_sim,v_error1[:,0]),
           (time_sim,v_error1[:,1]),
           (time_sim,v_error2[:,0]),
           (time_sim,v_error2[:,1])
           ]
    vxys = [vxys_1,
            vxys_2
           ]
    
    legends =["base (no noise)",
              "sim with a0",
              "w/noise correction",
              "w/noise correction + a0_learned"]
    
    fig_title   = ["TEST"]
    plot_xy(xys,legends =legends,fig_title =["Trajectories"]) 
    plot_traj(alphas,legends =['alpha_sim','alpha_sim_recovered',
                                      # 'alpha_cmd1',
                                      'alpha_cmd2'],fig_title =["alphas"])
    # plot_traj(vxs, legends = ['v_sim', 'v_cmd1','v_cmd2'],fig_title =["x-velocities"])
    # plot_traj(vys, legends = ['v_sim', 'v_cmd1','v_cmd2'],fig_title =["y-velocities"])
    # plot_traj(v_errs, legends = ['v_errorX1', 'v_errorY1','v_errorX2','v_errorY2'],fig_title =["v-errors"])
    # plot_vel([vxys_1],legends =['def a0'],fig_title =fig_title) 
    plot_vel([vxys_2],legends =['learned a0'],fig_title =fig_title) 
    break
# v_desired,v_error,v_stdv,vx,vy = vxys_1