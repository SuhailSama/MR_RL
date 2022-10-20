from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt


class LearningModule:
    def __init__(self):
        # kernel list is the kernel cookbook from scikit-learn
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 10.0)) + WhiteKernel()
        #create the X and Y GP regression objects
        self.gprX = GaussianProcessRegressor(kernel=kernel)
        self.gprY = GaussianProcessRegressor(kernel=kernel)

        self.X = []
        self.Yx = []
        self.Yy = []



    # px, py, alpha, time are numpy arrays, freq is constant
    # returns an estimate of a_0
    def learn(self, px, py, alpha, freq, time):
        #set time to start at 0
        time -= time[0]

        # apply smoothing to the position signals before calculating velocity 
        # dt is ~ 35 ms, so filter time ~= 0.035*N
        N = (int)(1 / 0.035) #filter position data due to noisy sensing

        px = uniform_filter1d(px, N, mode="nearest")
        py = uniform_filter1d(py, N, mode="nearest")

        #calculate velocity via position derivative
        vx = np.gradient(px, time)
        vy = np.gradient(py, time)

        # apply smoothing to the velocity signal
        vx = uniform_filter1d(vx, N, mode="nearest")
        vy = uniform_filter1d(vy, N, mode="nearest")

        #calculate speed to fit a_0
        speed = np.sqrt( vx**2 + vy**2 )

        #alpha  = 1k means the controller is off, delete those frames
        todel = np.argwhere(alpha >= 500)
        if len(todel) > 0:
            todel = int(todel[0])
            alpha = alpha[0:todel-1]
            px = px[0:todel-1]
            py = py[0:todel-1]
            vx = vx[0:todel-1]
            vy = vy[0:todel-1]
            time = time[0:todel-1]
            speed = speed[0:todel-1]
            if len(freq) >1:
                freq = freq[0:todel-1]

        a0 = np.median(speed) / freq 

        #generate empty NP arrays for X (data) and Y (outputs)
        X = alpha.reshape(-1,1)
        #v_e = v_actual - v_desired = v - a0*f*[ cos alpha; sin alpha]
        Yx = vx - a0 * freq * np.cos(alpha)
        Yy = vy - a0 * freq * np.sin(alpha)

        self.gprX.fit(X, Yx)
        self.gprY.fit(X, Yy)

        print("GP Learning Complete!")
        print("r^2 are " + str(self.gprX.score(X, Yx)) + " and " + str(self.gprY.score(X, Yy)) )

        self.X = X; self.Yx = Yx; self.Yy = Yy

        return a0



    def visualize(self):

        alpha_range = np.linspace( np.min(self.X), np.max(self.X), 200 )

        #evaluate the GPs
        muX,sigX = self.gprX.predict(alpha_range.reshape(-1, 1), return_std=True)
        muY,sigY = self.gprY.predict(alpha_range.reshape(-1, 1), return_std=True)

        #plot what the GP looks like for x velocity
        plt.figure()
        #plot pm 2 stdev
        plt.fill_between(alpha_range,  muX - 2*sigX,  muX + 2*sigX)
        plt.fill_between(alpha_range,  muX - sigX,  muX + sigX)
        #plot the data
        plt.plot(self.X, self.Yx, 'xk')
        #plot the approximate function
        plt.plot(alpha_range, muX, 'g')
        plt.title('X Axis Learning')
        plt.xlabel("alpha")
        plt.ylabel("V_e^x")

        #plot what the GP looks like for y velocity
        plt.figure()
        #plot pm 2 stdev
        plt.fill_between(alpha_range,  muY - 2*sigY,  muY + 2*sigY)
        plt.fill_between(alpha_range,  muY - sigY,  muY + sigY)
        #plot the data
        plt.plot(self.X, self.Yy, 'xk')
        #plot the approximate function
        plt.plot(alpha_range, muY, 'g')
        plt.title('Y Axis Learning')
        plt.xlabel("alpha")
        plt.ylabel("V_e^x")
        

        plt.show()




    def predict(self, alpha):

        muX,sigX = self.gprX.predict(alpha.reshape(1,-1), return_std=True)
        muY,sigY = self.gprY.predict(alpha.reshape(1,-1), return_std=True)

        return muX, muY, sigX, sigY

    '''
    #get bounds on learning - 2stdv ~= 95% of data
    plt.figure()
    plt.fill_between(time,   v_learned[:,0] - 2*gpX[:,1],   v_learned[:,0] + 2*gpX[:,1])
    plt.fill_between(time,   v_learned[:,0] -   gpX[:,1],   v_learned[:,0] +   gpX[:,1])

    plt.plot(time, v_learned[:,0], '-r', label="learned")
    plt.plot(time, v_desired[:,0], '-b', label="desired")
    plt.plot(time, vx, '-k', label="data")

    plt.legend()

    plt.show()

    '''





