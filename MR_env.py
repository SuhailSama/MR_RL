#!/usr/bin/python
#-*- coding: utf-8 -*-

from gym import Env, spaces
import numpy as np
from shapely.geometry import LineString, Point
from MR_viewer import Viewer
from MR_data import MRExperiment
from MR_simulator import Simulator
from matplotlib import animation
import matplotlib.pyplot as plt
import gym 
from typing import List, Tuple

REWARD_SUCCESS = 100
REWARD_FAILURE = -100
REWARD_STEP = -0.1

"""
TODO :
- Reward Function w/logical specs?
- 

"""
class MR_Env(Env):
    def __init__(
                self, 
                type = 'continuous', 
                action_dim = 2
                ):

        self.type = type
        self.action_dim = action_dim

        # assert type == 'continuous' or type == 'discrete', 'type must be continuous or discrete'
        # assert action_dim > 0 and action_dim <=2, 'action_dim must be 1 or 2'
        
        self.action_space = spaces.Box(
            low = np.array([0, 0]), 
            high = np.array([20, np.pi*2]))
        self.observation_space = spaces.Box(
            low = np.array([-5000, -5000, -5000, -5000, 0]), 
            high = np.array([5000, 5000, 5000, 5000, 80000])) # x,y,x_target,y_target,distance
        # Why these numbers?
        self.init_space = spaces.Box(
            low = np.array([100, 100]), 
            high = np.array([120, 120]))
        self.init_goal_space = spaces.Box(
            low = np.array([-31, -31]), 
            high = np.array([-32, -32]))
        self.borders = [
            [-510, 510],
            [-510, -510], 
            [510,-510], 
            [510, 510]]
        
        self.simulator = Simulator()
        
        self.last_loc = np.zeros(2)
        self.init_goal = np.zeros(2)
        self.last_action = np.zeros(self.action_dim)
        self.number_loop = 0  # loops in the screen -> used to plot
        self.counter = 0
        self.max_timesteps = 50
        self.min_dist2goal = 30
        
        self.MR_data = None
        self.name_experiment = None
        self.viewer = None
        self.test_performance = False
        # self.init_test_performance = np.linspace(0, np.pi / 15, 10)

    def step(
            self, 
            action: List[float]
            ) -> Tuple[List[float], float, bool]:
        """
        Returns:
            obs (object): an environment-specific object representing your observation of the environment: [x, y, x_target, y_target, distance to target]
            done (boolean): whether it’s time to reset the environment again.
        """
        # According to the action state a different kind of action is selected
        self.counter += 1
        f_t = action[0]
        alpha_t = action[1]
        state = self.simulator.step(f_t, alpha_t)

        obs = self.convert_state_to_observable(state, self.init_goal) 
        done = self.should_end(obs = obs)

        self.last_loc = [state[0], state[1]]
        self.last_action = np.array([f_t ,alpha_t])

        if self.MR_data is not None:
            # WHY? TODO
            reward = 10 # self.calculate_reward(obs=obs)
            self.MR_data.new_transition(state, obs, self.last_action, reward)
        
        return obs, done
    
    def convert_state_to_observable(
                                    self, 
                                    state: List[float, float], 
                                    goal_loc: List[float, float]
                                    ) -> List[float, float, float, float, float]:
        """
        This method generates the features used to build the reward function, converts the current state to the observable state
        Returns:
            obs (object): an environment-specific object representing your observation of the environment: [x, y, x_target, y_target, distance to target]
        """
        x, y, goal_x, goal_y  = state[0], state[1], goal_loc[0], goal_loc[1]
        
        cur_loc = np.array((x,y))
        goal_loc = np.array((goal_x,goal_y))
        d = np.linalg.norm( goal_loc - cur_loc )
        obs = np.array([x, y, goal_x, goal_y ,d])

        return obs

    def calculate_reward(
                        self, 
                        obs: List[float, float, float, float, float]
                        ) -> float:
        """
        This method calculates the reward based on the observable state
        Returns:
            reward (float) : amount of reward achieved by the previous action
        """
        d = obs[4]
        if d < self.min_dist2goal:
            print("\n ############ Got there ########")
            return REWARD_SUCCESS
        elif not self.observation_space.contains(obs) or self.counter > self.max_timesteps:
            return REWARD_FAILURE
        else:
            return REWARD_STEP # self.min_dist2goal/d

    def should_end(
                self, 
                obs: List[float, float, float, float, float]
                ) -> bool:
        """
        ? This method finds out whether we are at the end of episode
        """
        d = obs[4]
        if not self.observation_space.contains(obs) or self.counter > self.max_timesteps:
            # print("\n Smashed on wall")
            if self.viewer is not None:
                self.viewer.end_episode()
            if self.MR_data is not None:
                if self.MR_data.iterations > 0:
                    self.MR_data.save_experiment(self.name_experiment)
            return True
        elif d < self.min_dist2goal:
            return True
        else:
            return False

    def set_goal(
                self, 
                init
                ):
        """
        Needs to be changed??TODO
        """
        # self.init_goal = self.init_goal_space.sample()
        # while np.linalg.norm( self.init_goal - init ) < self.min_dist2goal :
        #     self.init_goal = self.init_space.sample()
        #     # print("uh")
        return self.init_goal

    def reset(
            self, 
            init: List[float, float] = None, 
            noise_var = 1, 
            a0 = 1, 
            is_mismatched = False
            ) -> List[float, float, float, float, float]:

        if init is None: 
            init = self.init_space.sample()
            
        print("starting positions")
        print(init.shape)
        self.set_goal(init)
        # Why these variables are not set at the init of the class? TODO
        # Because at the first iteration we are learning without noise?
        self.simulator.noise_var = noise_var
        self.simulator.a0 = a0
        self.simulator.reset_start_pos(init)
        self.simulator.is_mismatched = is_mismatched
        
        self.last_loc = init
        self.counter = 0
        # print('Reseting position')
        # print( "goal_loc ", init ,"init_pos",self.last_loc)
        state = self.simulator.get_state()
        if self.MR_data is not None:
            if self.MR_data.iterations > 0:
                self.MR_data.save_experiment(self.name_experiment)
            self.MR_data.new_iter(
                                state, 
                                self.convert_state_to_observable(state,self.init_goal), 
                                np.zeros(len(self.last_action)), 
                                np.array([0])
                                )
        if self.viewer is not None:
            self.viewer.end_episode()

        return self.convert_state_to_observable(
                                            state, 
                                            self.init_goal
                                            )

    def render(
            self, 
            mode = 'human'
            ) -> None:
        """
        Renders the environment if the mode is 'human'
        """
        if mode == 'human':
            if self.viewer is None:
                self.viewer = Viewer()
                self.viewer.plot_boundary(self.borders)
                
            if self.number_loop == 0:
                self.viewer.end_episode()
                self.viewer.plot_position(self.last_loc[0], self.last_loc[1])
                self.viewer.restart_plot()
                self.number_loop += 1
            else:
                self.viewer.plot_goal( self.init_goal, 2)
                self.viewer.plot_position(self.last_loc[0], self.last_loc[1])
                self.viewer.end_episode()
                self.viewer.restart_plot()


    def close(self) -> None:
        self.viewer.freeze_scream()

    # Unnecessary and unused
    def set_save_experice(self, name='experiment_ssn_ddpg_10iter'):
        assert type(name) == type(""), 'name must be a string'
        self.MR_data = MRExperiment()
        self.name_experiment = name

    def set_test_performace(self):
        self.test_performance = True


def save_frames_as_gif(
                    frames: List[np.ndarray], 
                    path='./', 
                    filename='gym_animation.gif'
                    ) -> None:

        #Mess with this to change frame size
        plt.figure(
                    figsize =(
                            frames[0].shape[1] / 72.0, 
                            frames[0].shape[0] / 72.0 ), 
                    dpi = 72
                    )

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
                                    plt.gcf(), 
                                    animate, 
                                    frames = len(frames), 
                                    interval=50)
        anim.save(
                path + filename, 
                writer = 'imagemagick', 
                fps = 60)

if __name__ == '__main__':
    frames = []
    mode = 'normal' # mode: 'normal', 'joystick'

    if mode == 'normal':
        env = MR_Env()
        episode_action_obs = []
        for i_episode in range(1):
            observation = env.reset()
            for t in range(20):
                frames.append(env.render())
                # env.render()
                action = np.array([20, np.pi/4]) # -2*np.pi/(i_episode+1)
                observation, done = env.step(action)
                
                # print ("observation, done, info \n")
                # print (observation)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()
        save_frames_as_gif(frames)
        print("######### DONE ########")

    elif mode == 'joystick':

        print('joystick not implemented')

    else:
        print("mode should be  'normal' or 'joystick'") 


