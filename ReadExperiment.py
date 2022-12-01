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
from scipy.interpolate import interp1d

folder = 'C:/Users/Logan/Downloads/Day5/Day5/test8BEST/'

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

h_de, = plt.plot(px_desired, py_desired, '--')
h_bl, = plt.plot(p_baseline[:,0], p_baseline[:,1])
h_lr, = plt.plot(p_learning[:,0], p_learning[:,1])

plt.legend([h_bl, h_lr, h_de], ['baseline', 'corrected', 'desired'])

plt.xlabel('x position (microns)')
plt.ylabel('y position (microns)')

plt.savefig('Day5-Test8-Trajectory.pdf', bbox_inches='tight')

plt.show()


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
plt.legend([h_bl, h_lr, h_de], ['baseline', 'corrected', 'desired'])
plt.ylabel('v_x (\mu/s)')
plt.xlabel('time (s)')

plt.show();


plt.figure()

h_de, = plt.plot(time_desired, vy_desired)
h_bl, = plt.plot(t_baseline, v_bl_y)
h_lr, = plt.plot(t_learning, v_lr_y)
plt.legend([h_bl, h_lr, h_de], ['baseline', 'corrected', 'desired'])
plt.ylabel('v_y (\mu/s)')
plt.xlabel('time (s)')

plt.show();

### Plot actual vs expected error
vx_function = interp1d(time_desired.squeeze(), vx_desired.squeeze(), kind='previous')

vx_bl_err = vx_function(t_baseline) - v_bl_x
vx_lr_err = vx_function(t_learning) - v_lr_x



plt.figure()

plt.plot(t_baseline, np.zeros(t_baseline.shape))
h1, = plt.plot(t_baseline, vx_bl_err)
h2, = plt.plot(t_learning, vx_lr_err)


plt.legend([h1, h2], ['baseline', 'corrected'])
plt.ylabel('v_e^x (\mu/s)')
plt.xlabel('time (s)')



vy_function = interp1d(time_desired.squeeze(), vy_desired.squeeze(), kind='previous')

vy_bl_err = vy_function(t_baseline) - v_bl_y
vy_lr_err = vy_function(t_learning) - v_lr_y



plt.figure()

plt.plot(t_baseline, np.zeros(t_baseline.shape))
h1, = plt.plot(t_baseline, vy_bl_err)
h2, = plt.plot(t_learning, vy_lr_err)

plt.legend([h1, h2], ['baseline', 'corrected'])
plt.ylabel('v_e^y (\mu/s)')
plt.xlabel('time (s)')


idxF = -1

print("\n RMSE Total:")

bl_rmse = np.sqrt( np.mean( vx_bl_err[0:idxF]**2 + vy_bl_err[0:idxF]**2  ) )
lr_rmse = np.sqrt( np.mean( vx_lr_err[0:idxF]**2 + vy_lr_err[0:idxF]**2  ) )

print("Baseline: " + str(bl_rmse))
print("Learning: " + str(lr_rmse))


bl_vx_rmse = np.sqrt( np.mean( vx_bl_err[0:idxF]**2) )
lr_vx_rmse = np.sqrt( np.mean( vx_lr_err[0:idxF]**2) )

bl_vy_rmse = np.sqrt( np.mean(vy_bl_err[0:idxF]**2  ) )
lr_vy_rmse = np.sqrt( np.mean(vy_lr_err[0:idxF]**2  ) )

print('\n RMSE Vx Vy:')

print("Baseline x: " + str(bl_vx_rmse))
print("Learning x: " + str(lr_vx_rmse))

print("Baseline y: " + str(bl_vy_rmse))
print("Learning y: " + str(lr_vy_rmse))

print('\nMean Error Vx Vy')

print("Mean BL Error: " + str(np.median(abs(vx_bl_err))) + ", " + str(np.median(abs(vy_bl_err))))
print("Mean LR Error: " + str(np.median(abs(vx_lr_err))) + ", " + str(np.median(abs(vy_lr_err))))

print("Percent Increase: " + str( (np.median(abs(vx_bl_err)) - np.median(abs(vx_lr_err))) / np.median(abs(vx_bl_err)) * 100 ) \
      + ", " + str( (np.median(abs(vy_bl_err)) - np.median(abs(vy_lr_err))) / np.median(abs(vy_bl_err)) * 100 ))

### look at the cumulative velocity error

cum_bl_err_x = integrate.cumulative_trapezoid(abs(vx_bl_err), t_baseline, initial=0)
cum_bl_err_y = integrate.cumulative_trapezoid(abs(vy_bl_err), t_baseline, initial=0)

cum_lr_err_x = integrate.cumulative_trapezoid(abs(vx_lr_err), t_learning, initial=0)
cum_lr_err_y = integrate.cumulative_trapezoid(abs(vy_lr_err), t_learning, initial=0)


#### Integrated Error
plt.figure()

plt.plot(t_baseline, np.zeros(t_baseline.shape))
h1, = plt.plot(t_baseline, np.abs(cum_bl_err_x))
h2, = plt.plot(t_learning, np.abs(cum_lr_err_x))

plt.legend([h1, h2], ['baseline', 'corrected'], loc='upper left')
plt.ylabel('Integrated Error (pixels)')
plt.xlabel('Time (s)')

plt.show()

plt.figure()

plt.plot(t_baseline, np.zeros(t_baseline.shape))
h1, = plt.plot(t_baseline, np.abs(cum_bl_err_y))
h2, = plt.plot(t_learning, np.abs(cum_lr_err_y))

plt.legend([h1, h2], ['baseline', 'corrected'], loc='upper left')
plt.ylabel('Integrated Error (pixels)')
plt.xlabel('Time (s)')

plt.show()


plt.figure()
hx, = plt.plot(t_baseline, np.abs(cum_bl_err_x) - np.abs(cum_lr_err_x))
hy, = plt.plot(t_baseline, np.abs(cum_bl_err_y) - np.abs(cum_lr_err_y))
plt.legend([hx, hy], ['x axis', 'y axis']) #, loc='upper left')
plt.ylabel('Cumulative Drift Improvement (pixels)')
plt.xlabel('Time (s)')

plt.savefig('cum-drift-improvement.pdf', bbox_inches='tight')


plt.show()


#calculate the 'mean drift' from the cumulative position error

bl_vx_drift = np.mean(abs(cum_bl_err_x))
bl_vy_drift = np.mean(abs(cum_bl_err_y))

lr_vx_drift = np.mean(abs(cum_lr_err_x))
lr_vy_drift = np.mean(abs(cum_lr_err_y))


print('\nDrift----')

print('Mean Baseline Drift: ' + str(bl_vx_drift) + ', ' + str(bl_vy_drift))
print('Mean Learning Drift: ' + str(lr_vx_drift) + ', ' + str(lr_vy_drift))

print('Mean Baseline Drift Improvement: ' + \
      str((bl_vx_drift-lr_vx_drift)/bl_vx_drift * 100) + ", " + \
          str((bl_vy_drift-lr_vy_drift)/bl_vy_drift * 100))

#### look at the final position error


pf_desired   = np.array([px_desired[-1], py_desired[-1]])
pf_baseline  = p_baseline[-1,:]
pf_learning  = p_learning[-1,:]


baseline_pf_err = np.sqrt( np.sum( (pf_baseline - pf_desired)**2 ) )
learning_pf_err = np.sqrt( np.sum( (pf_learning - pf_desired)**2 ) )


print("Baseline Pf Error: " + str(baseline_pf_err))
print("Learning Pf Error: " + str(learning_pf_err))

print('Percent improvement: ' + str( (baseline_pf_err - learning_pf_err)/baseline_pf_err * 100 ))




