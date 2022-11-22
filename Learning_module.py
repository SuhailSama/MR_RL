from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar


def objective(X, a0, freq, v_d, GPx, GPy, Dx, Dy):

    
    alpha = X
    
    X = np.array(alpha)
    
    mux = GPx.predict(X.reshape(1, -1))
    muy = GPy.predict(X.reshape(1, -1))


    #return (a0*freq*np.cos(alpha) + mux - v_d[0])**2 + (a0*freq*np.sin(alpha) + mux - v_d[1])**2

    return (a0*freq)**2 + (mux + Dx - v_d[0])**2 + 2*a0*freq*np.cos(alpha)*(mux + Dx - v_d[0]) \
                        + (muy + Dy - v_d[1])**2 + 2*a0*freq*np.sin(alpha)*(muy + Dy - v_d[1])


class LearningModule:
    def __init__(self):
        # kernel list is the kernel cookbook from scikit-learn
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel()
        #create the X and Y GP regression objects
        self.gprX = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gprY = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

        self.X = []
        self.Yx = []
        self.Yy = []

        self.a0 = 0
        self.f = 0
        self.Dx = 0
        self.Dy = 0



    def estimateDisturbance(self, px, py, time):
        N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing
        px = uniform_filter1d(px, N, mode="nearest")
        py = uniform_filter1d(py, N, mode="nearest")
        #calculate velocity via position derivative
        vx = np.gradient(px, time)
        vy = np.gradient(py, time)
        # apply smoothing to the velocity signal
        vx = uniform_filter1d(vx, (int)(N/2), mode="nearest")
        vy = uniform_filter1d(vy, (int)(N/2), mode="nearest")

        self.Dx = np.mean(vx[N:-N])
        self.Dy = np.mean(vy[N:-N])
        print("Estimated a D value of [" + str(self.Dx) + ", " + str(self.Dy) + "].")

    # px, py, alpha, time are numpy arrays, freq is constant
    # returns an estimate of a_0
    def learn(self, px, py, alpha_sim, time, actions):
        
        freq = actions[0,0]
        alpha = actions[:,1]
        time_action = actions[:,2] #we assume this starts at t=0
        
        #set time to start at 0
        time -= time[0]

        # apply smoothing to the position signals before calculating velocity 
        # dt is ~ 35 ms, so filter time ~= 0.035*N (this gives N = 38)
        N = (int)(1 / 0.035 / 2) #filter position data due to noisy sensing

        px = uniform_filter1d(px, N, mode="nearest")
        py = uniform_filter1d(py, N, mode="nearest")

        #calculate velocity via position derivative
        vx = np.gradient(px, time)
        vy = np.gradient(py, time)

        # apply smoothing to the velocity signal
        vx = uniform_filter1d(vx, (int)(N/2), mode="nearest")
        vy = uniform_filter1d(vy, (int)(N/2), mode="nearest")

        #calculate speed to fit a_0
        speed = np.sqrt( (vx - self.Dx )**2 + (vy - self.Dy)**2 )

        #alpha  = 1k means the controller is off, delete those frames
        todel = np.argwhere(alpha >= 500)
        if len(todel) > 0:
            todel = int(todel[0])
            alpha_sim = alpha_sim[0:todel-1]
            time_action = time_action[0:todel-1]
            px = px[0:todel-1]
            py = py[0:todel-1]
            vx = vx[0:todel-1]
            vy = vy[0:todel-1]
            time = time[0:todel-1]
            speed = speed[0:todel-1]

        #smoothing creates a boundary effect -- let's remove it
        alpha_sim = alpha_sim[N:-N]
        px = px[N:-N]
        py = py[N:-N]
        vx = vx[N:-N]
        vy = vy[N:-N]
        time = time[N:-N]
        speed = speed[N:-N]

        a0 = np.median(speed / freq)

        #generate empty NP arrays for X (data) and Y (outputs)
        #X = alpha.reshape(-1,1)
        
        X = np.array(alpha_sim).reshape(-1,1)
                
        #calculate the desired speed
        
   
        
        
        Yx = vx - a0 * freq * np.cos(alpha_sim)
        Yy = vy - a0 * freq * np.sin(alpha_sim)


        print(X.shape)
        print(Yx.shape)

        self.gprX.fit(X, Yx)
        self.gprY.fit(X, Yy)

        print("GP Learning Complete!")
        print("r^2 are " + str(self.gprX.score(X, Yx)) + " and " + str(self.gprY.score(X, Yy)) )

        
        a = np.linspace( np.min(X), np.max(X))
        
        Xe = a.reshape(-1, 1)
        
        e = self.gprX.predict(Xe)
        
        #plt.figure()
        #plt.plot(X, Yx, 'kx')
        #plt.plot(a, e, '-r')
        #plt.show()

        #plot the velocity error versus time
        #plt.figure()
        #plt.plot(time, vx, time, a0*freq*np.cos(alpha))
        #plt.show()


        self.X = X; self.Yx = Yx; self.Yy = Yy
        self.a0 = a0
        self.freq = freq

        return a0



    def visualize(self):

        alpha_range = np.linspace( np.min(self.X[:,0]), np.max(self.X[:,0]), 200 )
        
        

        X = alpha_range.reshape(-1, 1)

        #evaluate the GPs
        muX,sigX = self.gprX.predict(X, return_std=True)
        muY,sigY = self.gprY.predict(X, return_std=True)

        #plot what the GP looks like for x velocity
        plt.figure()
        #plot mean and stdv
        plt.fill_between(alpha_range, muX + 2*sigX, muX - 2*sigX)
        plt.fill_between(alpha_range, muX + sigX, muX - sigX)
        plt.plot(alpha_range, muX, 'r')
        #plot data points
        plt.plot(self.X, self.Yx, 'kx')
        
        plt.xlabel('alpha')
        plt.ylabel('v_e^x')
        
        plt.show()
        
        #plot what the GP looks like for y velocity
        plt.figure()
        #plot mean and stdv
        plt.fill_between(alpha_range, muY + 2*sigY, muY - 2*sigY)
        plt.fill_between(alpha_range, muY + sigY, muY - sigY)
        plt.plot(alpha_range, muY, 'r')
        #plot data points
        plt.plot(self.X, self.Yy, 'kx')
        
        plt.xlabel('alpha')
        plt.ylabel('v_e^y')
        
        plt.show()
      
        

    def error(self, vd):
        #alpha desired comes from arctan of desired velocity
        alpha_d = np.array(math.atan2(vd[1], vd[0]))
        
        X = np.array(alpha_d)
         
        #estimate the uncertainty for the desired alpha
        muX,sigX = self.gprX.predict(X.reshape(1,-1), return_std=True)
        muY,sigY = self.gprY.predict(X.reshape(1,-1), return_std=True)

        return muX, muY, sigX, sigY

    def predict(self, vd):
        #alpha desired comes from arctan of desired velocity
        alpha_d = np.array(math.atan2(vd[1], vd[0]))
        f_d = np.linalg.norm(vd) / self.a0

        X = np.array(alpha_d)
        

        #estimate the uncertainty for the desired alpha
        muX = self.gprX.predict(X.reshape(1,-1))
        muY = self.gprY.predict(X.reshape(1,-1))


        #select the initial alpha guess as atan2 of v_d - v_error
        x0 = np.array(alpha_d)
        
        
        #result = minimize(objective, x0, method='Powell', args=(self.a0, self.freq, vd, self.gprX, self.gprY), bounds=[(-np.pi, np.pi)])

        result = minimize_scalar(objective, method='Bounded', args=(self.a0, self.freq, vd, self.gprX, self.gprY, self.Dx, self.Dy), bounds=[-np.pi, np.pi] )


        X = np.array(result.x)

        #generate the uncertainty for the new alpha we're sending
        muX,sigX = self.gprX.predict(X.reshape(1,-1), return_std=True)
        muY,sigY = self.gprY.predict(X.reshape(1,-1), return_std=True)

        return X, muX, muY, sigX, sigY

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






