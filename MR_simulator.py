#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import RK45


class Simulator:
    def __init__(self):
        self.last_state = None
        self.current_action = None
        self.time_span = 1          #  seconds for each iteration
        self.number_iterations = 100  #  iterations for each step
        self.integrator = None # 
        ##MR Constants
        self.a = 1.6531366739638718
        self.state_prime = None

    def reset_start_pos(self, state_vector):
        x0, y0 = state_vector[0], state_vector[1]
        self.last_state = np.array([x0, y0])
        self.current_action = np.zeros(2)
        self.integrator = self.scipy_runge_kutta(self.simulate, self.get_state(), t_bound=self.time_span)

    def step(self, f_t, alpha_t):
        self.current_action = np.array([f_t, alpha_t])
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
        # print("\n States ",states)
        f_t = self.current_action[0]
        alpha_t = self.current_action[1]    
        # Derivative function

        # simple model
        mu, sigma = 0, 0.1 # mean and standard deviation
        dx1 = self.a * f_t  *np.cos(alpha_t) + np.random.normal(mu, sigma, 1)[0]
        dx2 = self.a * f_t  *np.sin(alpha_t) + np.random.normal(mu, sigma, 1)[0]

        # print("\n Actions taken:" , self.current_action)
        # print("\n np.cos(alpha_t) ",np.cos(alpha_t),"np.sin(alpha_t) ",np.sin(alpha_t))
        # print("\n dx1: ", dx1 , "dx2: ", dx2)
        fx = np.array([dx1, dx2])
        self.state_prime = fx
        return fx

    def scipy_runge_kutta(self, fun, y0, t0=0, t_bound=10):
        return RK45(fun, t0, y0, t_bound,  rtol=self.time_span/self.number_iterations, atol=1e-4)

    def get_state(self):
        return self.last_state

