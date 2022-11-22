# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:03:58 2022

@author: Logan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter1d
import scipy.integrate as integrate

folder = 'C:/Users/Logan/Downloads/Day5/Day5/test4/'

#with open(folder+'stats.pickle', 'rb') as file:
#    stats = pickle.load(file)
with open(folder+'step1.pickle', 'rb') as file: ##step 1 -- do nothing
    step1 = pickle.load(file)
with open(folder+'step2.pickle', 'rb') as file: ##step 2 -- learning
    step2 = pickle.load(file)
with open(folder+'step3.pickle', 'rb') as file: ##step 3 -- baseline
    step3 = pickle.load(file)
with open(folder+'step4.pickle', 'rb') as file: ##step 4 -- learned
    step4 = pickle.load(file)
    
print("unpickled the pickles")

step_idx = 0# this is usually 0, except when the file is weird


### see how the learning looks
p_baseline = np.array(step3[step_idx]['Positions'])
t_baseline = np.array(step3[step_idx]['Time'])

p_learning = np.array(step4[step_idx]['Positions'])
t_learning = np.array(step4[step_idx]['Time'])
### translate the curves to the origin
p_baseline = p_baseline - p_baseline[0,:]
p_learning = p_learning - p_learning[0,:]
### flip the curves
p_baseline = np.fliplr(p_baseline)
p_learning = np.fliplr(p_learning)
p_baseline[:,1] = -p_baseline[:,1]
p_learning[:,1] = -p_learning[:,1]
#set t0 = 0
t_baseline = t_baseline - t_baseline[0]
t_learning = t_learning - t_learning[0]



### extract the data from steps 1 and 2 to extract the disturbance and fit a0
p_idle = np.array(step1[0]['Positions'])
t_idle = np.array(step1[0]['Time'])

N = (int)(1 / 0.03) *2 #filter position data due to noisy sensing
p_idle_x = uniform_filter1d(p_idle[:,0], N, mode="nearest")
p_idle_y = uniform_filter1d(p_idle[:,1], N, mode="nearest")
#calculate velocity via position derivative
v_idle_x = np.gradient(p_idle_x, t_idle)
v_idle_y = np.gradient(p_idle_y, t_idle)
# apply smoothing to the velocity signal
v_idle_x = uniform_filter1d(v_idle_x, (int)(N/2), mode="nearest")
v_idle_y = uniform_filter1d(v_idle_y, (int)(N/2), mode="nearest")

Dx = np.mean(v_idle_x)
Dy = np.mean(v_idle_y)

print("D estimate: [" + str(Dx) + ", " + str(Dy) + "]")


p_test = np.array(step2[0]['Positions'])
t_test = np.array(step2[0]['Time'])

N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing
p_test_x = uniform_filter1d(p_test[:,0], N, mode="nearest")
p_test_y = uniform_filter1d(p_test[:,1], N, mode="nearest")
#calculate velocity via position derivative
v_test_x = np.gradient(p_test_x, t_test)
v_test_y = np.gradient(p_test_y, t_test)
# apply smoothing to the velocity signal
v_test_x = uniform_filter1d(v_test_x, (int)(N/2), mode="nearest")
v_test_y = uniform_filter1d(v_test_y, (int)(N/2), mode="nearest")

### extract the alphas from baseline (step3) to get the desired position
speed = np.sqrt( (v_test_x - Dx )**2 + (v_test_y - Dy)**2 )
a0 = np.median(speed / np.array(step2[0]['Frequency']))


speedX = np.sqrt( (v_test_x - Dx )**2 )
speedY = np.sqrt( (v_test_y - Dy)**2 )

a0x = np.median(speedX / np.array(step2[0]['Frequency']))
a0y = np.median(speedY / np.array(step2[0]['Frequency']))

print("a0 estimate is: " + str(a0))
print('a0 components: [' + str(a0x) + ", " + str(a0y) + ']')

### calculate the desired speed

alpha = np.array(step3[step_idx]['Alphas'])
freq  = np.array(step3[step_idx]['Frequency'])
time = np.array(step3[step_idx]['Time'])
time = time - time[0]

time_steps = 1000

dt = 0.03
time_desired = np.arange(0, dt*time_steps, dt).reshape(-1,1) #%np.array(step3[1]['Time'])


alpha_desired = np.zeros( (time_steps, 1) )

alpha_desired[0:200,0]   = np.linspace(0, np.pi/2, 200)
alpha_desired[200:400,0] = np.linspace(np.pi/2, -np.pi/2, 200)
alpha_desired[400:600,0] = np.linspace(-np.pi/2, 0, 200)
alpha_desired[600:800,0] = np.linspace(0, np.pi/8, 200)
alpha_desired[800::,0]  = np.linspace(np.pi/8, -np.pi, 200)


freq = np.median(freq)

vx_desired = (a0 * freq * np.cos(alpha_desired)).reshape(1,-1).flatten()
vy_desired = (a0 * freq * np.sin(alpha_desired)).reshape(1,-1).flatten()

px_desired = integrate.cumulative_trapezoid(vx_desired, dx=dt, initial=0)
py_desired = integrate.cumulative_trapezoid(vy_desired, dx=dt, initial=0)


## plot the results

plt.figure()

h_de, = plt.plot(px_desired, py_desired)
h_bl, = plt.plot(p_baseline[:,0], p_baseline[:,1])
h_lr, = plt.plot(p_learning[:,0], p_learning[:,1])

plt.legend([h_bl, h_lr, h_de], ['baseline', 'learning', 'desired'])

plt.show()


#N = (int)(1/0.03)

## find velocity curves
p_bl_x = uniform_filter1d(p_baseline[:,0], N, mode="nearest")
p_bl_y = uniform_filter1d(p_baseline[:,1], N, mode="nearest")
#calculate velocity via position derivative
v_bl_x = np.gradient(p_bl_x, t_baseline)
v_bl_y = np.gradient(p_bl_y, t_baseline)
# apply smoothing to the velocity signal
v_bl_x = uniform_filter1d(v_bl_x, (int)(N/2), mode="nearest")
v_bl_y = uniform_filter1d(v_bl_y, (int)(N/2), mode="nearest")


p_lr_x = uniform_filter1d(p_learning[:,0], N, mode="nearest")
p_lr_y = uniform_filter1d(p_learning[:,1], N, mode="nearest")
#calculate velocity via position derivative
v_lr_x = np.gradient(p_lr_x, t_learning)
v_lr_y = np.gradient(p_lr_y, t_learning)
# apply smoothing to the velocity signal
v_lr_x = uniform_filter1d(v_lr_x, (int)(N/2), mode="nearest")
v_lr_y = uniform_filter1d(v_lr_y, (int)(N/2), mode="nearest")


## plot velocity
plt.figure()

h_de, = plt.plot(time_desired, vx_desired)
h_bl, = plt.plot(t_baseline, v_bl_x)
h_lr, = plt.plot(t_learning, v_lr_x)
plt.legend([h_bl, h_lr, h_de], ['baseline', 'learning', 'desired'])
plt.ylabel('v_x')
plt.xlabel('time')

plt.show();


plt.figure()

h_de, = plt.plot(time_desired, vy_desired)
h_bl, = plt.plot(t_baseline, v_bl_y)
h_lr, = plt.plot(t_learning, v_lr_y)
plt.legend([h_bl, h_lr, h_de], ['baseline', 'learning', 'desired'])
plt.ylabel('v_y')
plt.xlabel('time')

plt.show();






