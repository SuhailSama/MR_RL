#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import RK45


class Simulator:
    def __init__(self):
        self.last_local_state = None
        self.current_action = None
        self.steps = 0
        self.time_span = 10           # 20 seconds for each iteration
        self.number_iterations = 100  # 100 iterations for each step
        self.integrator = None # 
        ##MR Constants
        self.a = 2

    def reset_start_pos(self, state_vector):
        x0, y0, vx0, vy0 = state_vector[0], state_vector[1], state_vector[2], state_vector[3]
        self.last_state = np.array([x0, y0, vx0, vy0])
        self.current_action = np.zeros(2)
        self.integrator = self.scipy_runge_kutta(self.simulate, self.get_state(), t_bound=self.time_span)

    def step(self, f_t, alpha_t):
        self.current_action = np.array([f_t, alpha_t])
        print("current_action: ", self.current_action)
        while not (self.integrator.status == 'finished'):
            self.integrator.step()
        self.last_state = self.integrator.y
        self.integrator = self.scipy_runge_kutta(self.simulate, self.get_state(), t0=self.integrator.t, t_bound=self.integrator.t+self.time_span)
        return self.last_state

    def simulate(self, t, states):
        """
        :param states: Space state
        :return df_states
        """
        x1 = states[0] #u
        x2 = states[1] #v
        x3 = states[2] #du
        x4 = states[3] #dv

        f_t = self.current_action[0]*np.pi/180  
        alpha_t = self.current_action[1]    

        # Derivative function
        fx1 = x3
        fx2 = x4

        # simple model
        fx3 = self.a * f_t  *np.cos(alpha_t)
        fx4 = self.a * f_t  *np.sin(alpha_t)

        fx = np.array([fx1, fx2, fx3, fx4])
        return fx 

    def scipy_runge_kutta(self, fun, y0, t0=0, t_bound=10):
        return RK45(fun, t0, y0, t_bound,  rtol=self.time_span/self.number_iterations, atol=1e-4)

    def get_state(self):
        return self.last_state

