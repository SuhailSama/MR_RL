#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import RK45
from typing import Tuple


class Simulator:
    def __init__(self):
        self.last_state = None
        self.current_action = None
        # as far as I understand we are using these numbers in main.py, too. To reduce confusion, I would suggest to define them in one place and import them in both files TODO
        self.time_span = 1.0/30         #  seconds for each iteration
        self.number_iterations = 100  #  iterations for each step
        self.integrator = None 
        ##MR Constants
        self.a0 = 0
        self.state_prime = None
        self.noise_var = 0
        self.is_mismatched = False

    def reset_start_pos(
                        self, 
                        state_vector: Tuple[float, float]
                        ) -> None:
        """
        Reset the start position of the simulator
        """

        x0, y0 = state_vector[0], state_vector[1]
        self.last_state = np.array([x0, y0])
        self.current_action = np.zeros(2)
        self.integrator = self.scipy_runge_kutta(self.simulate, self.get_state(), t_bound=self.time_span)

    def step(
            self, 
            f_t: float, 
            alpha_t: float
            ) -> Tuple[float, float]:
        """
        Step the simulator by one time step
        """

        self.current_action = np.array([f_t, alpha_t])
        while not (self.integrator.status == 'finished'):
            self.integrator.step()
        
        self.last_state = self.integrator.y
        self.integrator = self.scipy_runge_kutta(
                                            self.simulate, 
                                            self.get_state(), 
                                            t0 = self.integrator.t, 
                                            t_bound = self.integrator.t + self.time_span)

        return self.last_state


    def a0_linear(self, alpha_t, sigma):
        return self.a0 + (alpha_t/np.pi)*0.4 + np.random.normal(0, sigma, 1)[0]

    def simulate(
                self, 
                t, 
                states: Tuple[float, float]
                ) -> Tuple[float, float]:
        # print("\n States ",states)
        f_t = self.current_action[0]
        alpha_t = self.current_action[1]

        # simple model
        mu, sigma = 0, self.noise_var # mean and standard deviation

        #select a value of a0 -- either costant or with model mismatch
        a0 = self.a0
        if self.is_mismatched:
            #a0 = self.a0_linear(alpha_t, sigma/4)
            dx1 = a0 * f_t  * np.cos(alpha_t + 0.1) + np.random.normal(mu, sigma, 1)[0] + 0.2
            dx2 = a0 * f_t  * np.sin(alpha_t + 0.1) + np.random.normal(mu, sigma, 1)[0] - 0.1
        else:
            dx1 = a0 * f_t  * np.cos(alpha_t) + np.random.normal(mu, sigma, 1)[0] 
            dx2 = a0 * f_t  * np.sin(alpha_t) + np.random.normal(mu, sigma, 1)[0] 

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

