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

folder = 'test5_4x/'

with open(folder+'stats.pickle', 'rb') as file:
    stats = pickle.load(file)
with open(folder+'step1.pickle', 'rb') as file: ##step 1 -- do nothing
    step1 = pickle.load(file)
with open(folder+'step2.pickle', 'rb') as file: ##step 2 -- learning
    step2 = pickle.load(file)
with open(folder+'step3.pickle', 'rb') as file: ##step 3 -- baseline
    step3 = pickle.load(file)
with open(folder+'step4.pickle', 'rb') as file: ##step 4 -- learned
    step4 = pickle.load(file)
    
print("unpickled the pickles")


### see how the learning looks
p_baseline = np.array(step3[0]['Positions'])
p_learning = np.array(step4[0]['Positions'])

### translate the curves to the origin
p_baseline = p_baseline - p_baseline[0,:]
p_learning = p_learning - p_learning[0,:]
### flip the curves
p_baseline = np.fliplr(p_baseline)
p_learning = np.fliplr(p_learning)
p_baseline[:,1] = -p_baseline[:,1]
p_learning[:,1] = -p_learning[:,1]


### extract the data from steps 1 and 2 to extract the disturbance and fit a0
p_idle = np.array(step1[0]['Positions'])
t_idle = np.array(step1[0]['Time'])

N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing
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

alpha = np.array(step3[0]['Alphas'])
freq  = np.array(step3[0]['Frequency'])
time  = np.array(step3[0]['Time'])

vx_desired = a0 * freq * np.cos(alpha)
vy_desired = a0 * freq * np.sin(alpha)

px_desired = integrate.cumulative_trapezoid(vx_desired, time)
py_desired = integrate.cumulative_trapezoid(vy_desired, time)


## plot the results

plt.figure()

h_de, = plt.plot(px_desired, py_desired)
h_bl, = plt.plot(p_baseline[:,0], p_baseline[:,1])
h_lr, = plt.plot(p_learning[:,0], p_learning[:,1])

plt.legend([h_bl, h_lr, h_de], ['baseline', 'learning', 'desired'])

plt.show()