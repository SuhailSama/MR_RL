import numpy as np
import Learning_module as GP # type: ignore
from utils import run_sim,find_alpha_corrected
from utils import plot_xy,plot_traj,plot_bounded_curves
from scipy.ndimage import uniform_filter1d


#frequency of the magnetic field in Hz and the nominal value of a0
freq = 4
a0_def = 1.5
dt = 0.030 #assume a timestep of 30 ms


#first we do nothing
time_steps = 100 #do nothing for 100/30 seconds
actions_idle = np.zeros((time_steps, 3))
actions_idle[:,2] = np.arange(0, dt*time_steps, dt)


#note: timestep is 1/30 seconds, the rate we get data at in the experiment
time_steps = 1800 #train for 10s at 30 hz
cycles = 3 #train my moving in 3 circles

steps = (int)(time_steps / cycles)

#generate actions to move in a circle at a constant frequency
actions_circle = np.zeros( (steps, 3))
actions_circle[:,0] = freq
actions_circle[:,1] = np.linspace(-np.pi, np.pi, steps)

#stack the circle actions to get our learning set
actions_learn = np.vstack([actions_circle]*cycles)
actions_learn[:,2] = np.arange(0, dt*time_steps, dt)

t = np.linspace(0, time_steps, time_steps)


#generate actions for testing (1/30 hz for 30 seconds)
time_steps = 1000

actions = np.zeros( (time_steps, 3) )

actions[0:200,1]   = np.linspace(0, np.pi/2, 200)
actions[200:400,1] = np.linspace(np.pi/2, -np.pi/2, 200)
actions[400:600,1] = np.linspace(-np.pi/2, 0, 200)
actions[600:800,1] = np.linspace(0, np.pi/8, 200)
actions[800::,1]  = np.linspace(np.pi/8, -np.pi, 200)

actions[:,0] = freq # np.linspace(3, 4, time_steps)
actions[:,2] = np.arange(0, dt*time_steps, dt)


noise_var = 0.5


gp_sim = GP.LearningModule()

#first we will do absolutely nothing to try and calculate the drift term
px_idle,py_idle,alpha_idle,time_idle,freq_idle = run_sim(actions_idle,
                                                            init_pos = np.array([0,0]),
                                                            noise_var = noise_var,
                                                            a0=a0_def,is_mismatched=True)


gp_sim.estimateDisturbance(px_idle, py_idle, time_idle)


# THIS IS WHAT THE SIMULATION ACTUALLY GIVES US -- model mismatch && noise
px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions_learn,
                                                     init_pos = np.array([0,0]),
                                                     noise_var = noise_var,
                                                     a0=a0_def,is_mismatched=True)


# learn noise and a0 -- note px_desired and py_desired need to be at the same time
a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim, time_sim, actions)
print("Estimated a0 value is " + str(a0_sim))
gp_sim.visualize()


# THIS CALCULATES THE DESIRED TRAJECTORY FROM OUR a0 ESTIMATE
px_desired,py_desired,alpha_desired,time_desired,freq_desired = run_sim(actions_learn,
                                                                        init_pos = np.array([0,0]),
                                                                        noise_var = 0.0,a0=a0_sim)

# plot the desired vs achieved velocities
xys  = [(px_desired,py_desired),
        (px_sim,py_sim),
       ]
legends =["Desired Trajectory","Simulated Trajectory (no learning)"
          ]
fig_title   = ["Learning Dataset"]
plot_xy(xys,legends =legends,fig_title =fig_title) 


###### END OF LEARNING, NOW WE DO TESTING ######


# Desired Trajectory: no noise, no learning -- this is the desired trajectory
px_desired,py_desired,alpha_desired,time_desired,freq_desired = run_sim(actions,
                                                         init_pos = np.array([0,0]),
                                                         noise_var = 0.0,
                                                         a0=a0_sim) #assume we used a0_sim to generate the control actions

# Baseline: actual noise and parameters, no learning -- this is the achieved trajectory
px_baseline,py_baseline,alpha_baseline, time_baseline,freq_baseline = run_sim(actions,
                                                     init_pos = np.array([0,0]),
                                                     noise_var = noise_var,
                                                     a0=a0_def,is_mismatched=True)

#generate our desired, predicted, and error bars for velocity for the test
vd = np.zeros( (len(actions), 2) )
v_pred = np.zeros( (len(actions), 2) )
v_stdv  = np.zeros( (len(actions), 2) )
actions_corrected = np.zeros(actions.shape)

for ii in range(len(actions_corrected)):
    vd[ii,:] = a0_sim*freq*np.array( [np.cos(actions[ii,1]), np.sin(actions[ii,1])] ).reshape(1,-1)
    #actions_corrected[ii,0] = actions[ii,0] #don't correct the rolling frequency
    A, muX, muY, sigX, sigY = find_alpha_corrected(vd[ii],gp_sim)
    
    actions_corrected[ii,0] = actions[ii,0]
    actions_corrected[ii,1] = A
    
    #our predicted velocity is model + error
    v_pred[ii,0] = a0_sim*freq*np.cos(actions_corrected[ii,1]) + muX
    v_pred[ii,1] = a0_sim*freq*np.sin(actions_corrected[ii,1]) + muY
    v_stdv[ii,0] = sigX
    v_stdv[ii,1] = sigY
    

 # sim: noise, learning
px_learn,py_learn,alpha_learn, time_learn,freq_learn = run_sim(actions_corrected,
                                                     init_pos = np.array([0,0]),
                                                     noise_var = noise_var,
                                                     a0=a0_def,is_mismatched=True) #simulate using the true value of a0


#### Plot Resulting Trajectories
xys = [(px_desired,py_desired),
       (px_baseline,py_baseline),
       (px_learn,py_learn)]
legends= ["desired",
            "baseline",
            "corrected"]
plot_xy(xys,legends =legends,savefile_name='sim_pos.pdf') 


alphas = [(time_baseline,actions[:,1]),
          (time_learn,actions_corrected[:,1]) ]

plot_traj(alphas,legends =['alpha',
                                  'alpha_corrected'],fig_title =["alphas"])


### plot x and y velocity bounds
N = (int)(1 / dt / 2 )

px_learn = uniform_filter1d(px_learn, N, mode="nearest")
px_learn = uniform_filter1d(px_learn, N, mode="nearest")
vx_learn = np.gradient(px_learn, time_learn)
vy_learn = np.gradient(py_learn, time_learn)
vx_learn = uniform_filter1d(vx_learn, (int)(N/2), mode="nearest")
vy_learn = uniform_filter1d(vy_learn, (int)(N/2), mode="nearest")

vx_baseline = np.gradient(px_baseline, time_baseline)
vy_baseline = np.gradient(py_baseline, time_baseline)
vx_baseline = uniform_filter1d(vx_baseline, (int)(N/2), mode="nearest")
vy_baseline = uniform_filter1d(vy_baseline, (int)(N/2), mode="nearest")


vx_curve = [(time_learn,      vx_learn),
            (time_baseline[N:-N], vx_baseline[N:-N]),
            (time_desired,  a0_def*freq*np.cos(alpha_desired))]
vx_bounds   = []
plot_bounded_curves(vx_curve,vx_bounds,legends=['learning', 'uncorrected', 'desired'], fig_title=["Vx Profile"])

vy_curve = [(time_learn,    vy_learn),
            (time_baseline[N:-N], vy_baseline[N:-N]),
            (time_desired,  a0_def*freq*np.sin(alpha_desired))]
vy_bounds   = []
plot_bounded_curves(vy_curve,vy_bounds,legends=['learning', 'uncorrected', 'desired'], fig_title=["Vy Profile"])



###plot the desired vs actual velocity with the GP bounds -- see if we learned the error or not
vx_desired = a0_sim * freq * np.cos(alpha_desired)
vy_desired = a0_sim * freq * np.sin(alpha_desired)

vel_error = np.zeros( (len(time_desired), 2) )
vel_sigma = np.zeros( (len(time_desired), 2) )
for ti in range(len(time_desired)):
    muX, muY, sigX, sigY = gp_sim.error( [vx_desired[ti], vy_desired[ti]] )
    
    vel_error[ti,:] = np.array([muX, muY]).reshape(1,-1)

    vel_sigma[ti,:] = np.array([sigX, sigY]).reshape(1,-1)



vx_curve = [(time_baseline, vx_baseline),
            (time_desired,  vx_desired),
            (time_desired,  vx_desired + vel_error[:,0])]
vx_bounds   = [(time_desired, vx_desired + vel_error[:,0] + 2*vel_sigma[:,0], vx_desired + vel_error[:,0] - 2*vel_sigma[:,0]), 
               (time_desired, vx_desired + vel_error[:,0] + vel_sigma[:,0], vx_desired + vel_error[:,0] - vel_sigma[:,0])]

plot_bounded_curves(vx_curve,vx_bounds,legends=['baseline', 'desired', 'estimate'], fig_title=["Estimating Vx Error"])


vy_curve = [(time_baseline, vy_baseline),
            (time_desired,  vy_desired),
            (time_desired,  vy_desired + vel_error[:,1])]
vy_bounds   = [(time_desired, vy_desired + vel_error[:,1] + 2*vel_sigma[:,1], vy_desired + vel_error[:,1] - 2*vel_sigma[:,1]), 
               (time_desired, vy_desired + vel_error[:,1] + vel_sigma[:,1], vy_desired + vel_error[:,1] - vel_sigma[:,1])]

plot_bounded_curves(vy_curve,vy_bounds,legends=['baseline', 'desired', 'estimate'], fig_title=["Estimating Vy Error"])


### better error estimate plots

vx_error = vx_baseline - vx_desired

error_curve = [(time_baseline, vx_error)]
error_bound = [(time_desired, vel_error[:,0] + 2*vel_sigma[:,0], vel_error[:,0] - 2*vel_sigma[:,0]), 
               (time_desired, vel_error[:,0] + vel_sigma[:,0],   vel_error[:,0] - vel_sigma[:,0])]    
plot_bounded_curves(error_curve,error_bound,savefile_name='Vx_estimate.pdf')


vy_error = vy_baseline - vy_desired

error_curve = [(time_baseline, vy_error)]
error_bound = [(time_desired, vel_error[:,1] + 2*vel_sigma[:,1], vel_error[:,1] - 2*vel_sigma[:,1]), 
               (time_desired, vel_error[:,1]   + vel_sigma[:,1], vel_error[:,1] -   vel_sigma[:,1])]    
plot_bounded_curves(error_curve,error_bound,savefile_name='Vy_estimate.pdf')
