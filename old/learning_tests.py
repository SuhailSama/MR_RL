import math
import pickle
from random import uniform

import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

from scipy.ndimage import uniform_filter1d

from datetime import datetime

#extract the data
with open('closedloopdata-10-1_withTIME/closed.pickle', 'rb') as f:
#with open('closed_loopdata_9-26/test1.pickle', 'rb') as f:
    data = pickle.load(f)

dataset = data[0]
params = dataset['Track_Params(frame,error,current_pos,target_pos,alpha,time)']

freq = data[2]

#get alpha from the parameters
px = []
py = []
tx = []
ty = []
alpha = []
time = []

for i in range(0, len(params)):
    row = params[i]
    px.append(row[2][0])
    py.append(row[2][1])
    tx.append(row[3][0])
    ty.append(row[3][1])
    alpha.append(row[4])
    time.append(row[5])
    


print('Finished loading pickle file\n')

#get px, py, vx, vy, alpha as numpy arrays
px = np.array(px)
py = np.array(py)
tx = np.array(tx)
ty = np.array(ty)
alpha = np.array(alpha)
time = np.array(time)
time -= time[0]

# apply smoothing to the position signals before calculating velocity 
# dt is ~ 35 ms, so filter time ~= 0.035*N
N = (int)(1 / 0.035) #filter position data due to noisy sensing

px = uniform_filter1d(px, N, mode="nearest")
py = uniform_filter1d(py, N, mode="nearest")

vx = np.gradient(px, time)
vy = np.gradient(py, time)

#filter velocity calculation caused by numerical noise
vx = uniform_filter1d(vx, N, mode="nearest")
vy = uniform_filter1d(vy, N, mode="nearest")

speed = np.sqrt( vx**2 + vy**2 )

#alpha  = 1k means the controller is off, delete those frames
todel = np.argwhere(alpha >= 500)
if len(todel) > 0:
    todel = int(todel[0])
    alpha = alpha[0:todel-1]
    tx = tx[0:todel-1]
    ty = ty[0:todel-1]
    px = px[0:todel-1]
    py = py[0:todel-1]
    vx = vx[0:todel-1]
    vy = vy[0:todel-1]
    time = time[0:todel-1]
    speed = speed[0:todel-1]



#alpha saved w.r.t. the x axis, but it should be w.r.t. the y axis
#alpha = np.pi/2 - alpha

numFrames = len(alpha)


#plt.plot(px, py)
#plt.quiver(px, py, vx, vy)
#plt.show()

#step 1: use regression to estimate a_0 assuming E[D]=0 from the data
# |v| = |a_0 f(t)| -> |a_0| = |v| / |f(t)|
a0 = np.median(speed) / freq # max uses 1 np.median(speed) / freq
print('mean speed: ' + str(np.median(speed)))
print('a0 estimate is ' + str(a0))


#plt.plot(dataset['VMag'])
#plt.plot(speed)
#plt.show()


#step 2: take the desired and actual velocity

#dot and white is a noisy linear regression
#kernel = DotProduct() + WhiteKernel()

#These have fit issues, need to change lower bounds to get convergence.
    # this likely means that some feature (e.g., periodicity) should be ignored
#kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
#kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
#kernel = ExpSineSquared(length_scale=1, periodicity=1)


# kernel list is the kernel cookbook from scikit-learn
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel() # + DotProduct() + WhiteKernel()



gprX = GaussianProcessRegressor(kernel=kernel)
gprY = GaussianProcessRegressor(kernel=kernel)


# measure v_e and learn it as a function of p, f, \alpha.
stepSize = 5
#calculate size of arrays for num of points we will learn
numLearnedPts = math.ceil(numFrames / stepSize)
# in this data f is a constant value of 4, so learn v_e = f(p, \alpha)
X  = np.zeros( (numLearnedPts, 1) )
Yx = np.zeros( (numLearnedPts, 1) )
Yy = np.zeros( (numLearnedPts, 1) )


print("Total Frames: " + str(numFrames))
print("Learning with: " + str(numLearnedPts))



ctr = 0
for i in range(0, numFrames, stepSize):
    v_desired = a0*freq*np.array( [np.cos(alpha[i]), np.sin(alpha[i])] )
    v_actual = np.array( [vx[i], vy[i]] )
    v_e = v_actual - v_desired

  #  X[ctr,0] = px[i]
  #  X[ctr,1] = py[i]
    X[ctr,0] = alpha[i]
    #X[ctr,1] = (time[i] * freq) - math.floor(time[i] * freq) #percent of a rotation
    #.append( [ float(px[i]), float(py[i]), float(alpha[i]) ], 2 )
    Yx[ctr] = v_e[0]
    Yy[ctr] = v_e[1]
    #Yx.append(float(v_e[0]))
    #Yy.append(float(v_e[1]))

    ctr += 1


start=datetime.now()

gprX.fit(X, Yx)
gprY.fit(X, Yy)

print( datetime.now()-start )

print( gprX.score(X, Yx) )


a_vals = np.linspace(np.min(X), np.max(X), 500)
muX,sigX = gprX.predict(a_vals.reshape(-1, 1), return_std=True)


#plot what the GP looks like for x velocity
plt.figure()
#plot pm 2 stdev
plt.fill_between(a_vals,  muX - 2*sigX,  muX + 2*sigX)
plt.fill_between(a_vals,  muX - sigX,  muX + sigX)
#plot the data
plt.plot(X, Yx, 'xk')
#plot the approximate function
plt.plot(a_vals, muX, 'g')

plt.xlabel("alpha")
plt.ylabel("V_e^x")
#plt.show()


v_learned = np.zeros( (numFrames, 2) )
v_desired = np.zeros( (numFrames, 2) )

gpX = np.zeros( (numFrames, 2) )
gpY = np.zeros( (numFrames, 2) )

for i in range(0, numFrames):
    v_desired[i,:] = a0*freq*np.array( [np.cos(alpha[i]), np.sin(alpha[i])] )

    #Xp = np.array( [ float(px[i]), float(py[i]), float(alpha[i]), float(time[i]/freq) - math.floor(time[i] / freq) ] )
    Xp = np.array( [float(alpha[i])] ) #, float(time[i] * freq) - math.floor(time[i] * freq) ] )

    Xp = Xp.reshape(1,-1) #correct size for GP to predict

    muX,sigX = gprX.predict(Xp, return_std=True)
    muY,sigY = gprY.predict(Xp, return_std=True)

    gpX[i,0] = muX; gpX[i,1] = sigX
    gpY[i,0] = muY; gpY[i,1] = sigY

    v_learned[i,0] = v_desired[i,0] + gpX[i,0]
    v_learned[i,1] = v_desired[i,1] + gpY[i,0]

#get bounds on learning - 2stdv ~= 95% of data
plt.figure()
plt.fill_between(time,   v_learned[:,0] - 2*gpX[:,1],   v_learned[:,0] + 2*gpX[:,1])
plt.fill_between(time,   v_learned[:,0] -   gpX[:,1],   v_learned[:,0] +   gpX[:,1])

plt.plot(time, v_learned[:,0], '-r', label="learned")
plt.plot(time, v_desired[:,0], '-b', label="desired")
plt.plot(time, vx, '-k', label="data")

plt.legend()

plt.show()








