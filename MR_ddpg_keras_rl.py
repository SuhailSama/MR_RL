import numpy as np
import pickle
import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda
from keras.models import Sequential, Model
from tensorflow.keras import layers

from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

from MR_env import MR_Env
# from RL import OUActionNoise, Buffer


from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.layers import Dense, Activation, Flatten, Input, Concatenate


# Get the environment and extract the number of actions.
env = MR_Env()

# problem = "Pendulum-v0"
# env = gym.make(problem)

# np.random.seed(555)
# env.seed(555)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]



num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# ______________________________________
# Next, we build a very simple model. (https://keras.io/examples/rl/ddpg_pendulum/)
# Initialize weights between -3e-3 and 3-e3
last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(36, kernel_initializer=last_init))
actor.add(Activation('relu'))
actor.add(Dense(18, kernel_initializer=last_init))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions, kernel_initializer=last_init))
actor.add(Activation('softsign'))
# actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(36, kernel_initializer=last_init)(x)
x = Activation('relu')(x)
x = Dense(18, kernel_initializer=last_init)(x)
x = Activation('relu')(x)
x = Dense(1, kernel_initializer=last_init)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.6, mu=0, sigma=0.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, batch_size=500, nb_steps_warmup_critic=2000, nb_steps_warmup_actor=1000,
                   gamma=.99,random_process=random_process, target_model_update=5e-2) #
agent.compile(Adam(lr=0.1,  clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something!
mode = 'train'
if mode == 'train':
    hist = agent.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=None)
    filename = '600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'
    # # we save the history of learning, it can further be used to plot reward evolution
    with open('_experiments/history_ddpg__redetorcs'+filename+'.pickle', 'wb') as handle:
         pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # After training is done, we save the final weights.
    agent.save_weights('h5f_files/ddpg_{}_weights.h5f'.format('600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2_action_lim_1'), overwrite=True)

    # # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=100, visualize=True, nb_max_episode_steps=None)

elif mode == 'test':
    env.set_test_performace() # Define the initialization as performance test
    env.set_save_experice()   # Save the test to plot the results after
    agent.load_weights('h5f_files/ddpg_{}_weights.h5f'.format('600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2_action_lim_1'))
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=None)
