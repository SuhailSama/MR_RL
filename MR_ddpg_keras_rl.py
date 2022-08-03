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


# tf.enable_eager_execution() 
# class OUActionNoise:
#     def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
#         self.theta = theta
#         self.mean = mean
#         self.std_dev = std_deviation
#         self.dt = dt
#         self.x_initial = x_initial
#         self.reset()

#     def __call__(self):
#         # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
#         x = (
#             self.x_prev
#             + self.theta * (self.mean - self.x_prev) * self.dt
#             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
#         )
#         # Store x into x_prev
#         # Makes next noise dependent on current one
#         self.x_prev = x
#         return x

#     def reset(self):
#         if self.x_initial is not None:
#             self.x_prev = self.x_initial
#         else:
#             self.x_prev = np.zeros_like(self.mean)


# class Buffer:
#     def __init__(self, buffer_capacity=100000, batch_size=64, num_states = 2 ,num_actions = 2):
#         # Number of "experiences" to store at max
#         self.buffer_capacity = buffer_capacity
#         # Num of tuples to train on.
#         self.batch_size = batch_size

#         # Its tells us num of times record() was called.
#         self.buffer_counter = 0

#         # Instead of list of tuples as the exp.replay concept go
#         # We use different np.arrays for each tuple element
#         self.state_buffer = np.zeros((self.buffer_capacity, num_states))
#         self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
#         self.reward_buffer = np.zeros((self.buffer_capacity, 1))
#         self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

#     # Takes (s,a,r,s') obervation tuple as input
#     def record(self, obs_tuple):
#         # Set index to zero if buffer_capacity is exceeded,
#         # replacing old records
#         index = self.buffer_counter % self.buffer_capacity

#         self.state_buffer[index] = obs_tuple[0]
#         self.action_buffer[index] = obs_tuple[1]
#         self.reward_buffer[index] = obs_tuple[2]
#         self.next_state_buffer[index] = obs_tuple[3]

#         self.buffer_counter += 1

#     # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
#     # TensorFlow to build a static graph out of the logic and computations in our function.
#     # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
#     @tf.function
#     def update(
#         self, state_batch, action_batch, reward_batch, next_state_batch,
#     ):
#         # Training and updating Actor & Critic networks.
#         # See Pseudo Code.
#         with tf.GradientTape() as tape:
#             target_actions = target_actor(next_state_batch, training=True)
#             y = reward_batch + gamma * target_critic(
#                 [next_state_batch, target_actions], training=True
#             )
#             critic_value = critic_model([state_batch, action_batch], training=True)
#             critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

#         critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
#         critic_optimizer.apply_gradients(
#             zip(critic_grad, critic_model.trainable_variables)
#         )

#         with tf.GradientTape() as tape:
#             actions = actor_model(state_batch, training=True)
#             critic_value = critic_model([state_batch, actions], training=True)
#             # Used `-value` as we want to maximize the value given
#             # by the critic for our actions
#             actor_loss = -tf.math.reduce_mean(critic_value)

#         actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
#         actor_optimizer.apply_gradients(
#             zip(actor_grad, actor_model.trainable_variables)
#         )

#     # We compute the loss and update parameters
#     def learn(self):
#         # Get sampling range
#         record_range = min(self.buffer_counter, self.buffer_capacity)
#         # Randomly sample indices
#         batch_indices = np.random.choice(record_range, self.batch_size)

#         # Convert to tensors
#         state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
#         action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
#         reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
#         reward_batch = tf.cast(reward_batch, dtype=tf.float32)
#         next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

#         self.update(state_batch, action_batch, reward_batch, next_state_batch)

# # This update target parameters slowly
# # Based on rate `tau`, which is much less than one.
# @tf.function
# def update_target(target_weights, weights, tau):
#     for (a, b) in zip(target_weights, weights):
#         a.assign(b * tau + a * (1 - tau))


# def get_actor():
#     # Initialize weights between -3e-3 and 3-e3
#     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

#     inputs = layers.Input(shape=(num_states,))
#     out = layers.Dense(256, activation="relu", kernel_initializer=last_init)(inputs)
#     out = layers.Dense(256, activation="relu", kernel_initializer=last_init)(out)
#     outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

#     # Our upper bound is 2.0 for Pendulum.
#     outputs = outputs
#     model = tf.keras.Model(inputs, outputs)
#     return model


# def get_critic():
#     last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)

#     # State as input
#     state_input = layers.Input(shape=(num_states))
#     state_out = layers.Dense(16, activation="relu", kernel_initializer=last_init)(state_input)
#     state_out = layers.Dense(32, activation="relu", kernel_initializer=last_init)(state_out)

#     # Action as input
#     action_input = layers.Input(shape=(num_actions))
#     action_out = layers.Dense(32, activation="relu", kernel_initializer=last_init)(action_input)

#     # Both are passed through seperate layer before concatenating
#     concat = layers.Concatenate()([state_out, action_out])

#     out = layers.Dense(256, activation="relu", kernel_initializer=last_init)(concat)
#     out = layers.Dense(256, activation="relu", kernel_initializer=last_init)(out)
#     outputs = layers.Dense(1)(out)

#     # Outputs single value for give state-action
#     model = tf.keras.Model([state_input, action_input], outputs)

#     return model

# def policy(state, noise_object):
#     sampled_actions = tf.squeeze(actor_model(state))
#     noise = noise_object()
#     # Adding noise to action
#     sampled_actions = sampled_actions.numpy() + noise
#     sampled_actions = sampled_actions * upper_bound
#     # We make sure action is within bounds
#     legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
#     # print (" \n sampled_actions:",sampled_actions)
#     return np.squeeze(legal_action)


# # Training hyperparameters
# std_dev = 0.01
# ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# actor_model = get_actor()
# critic_model = get_critic()

# target_actor = get_actor()
# target_critic = get_critic()

# # Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# # Learning rate for actor-critic models
# critic_lr = 0.002
# actor_lr = 0.001

# critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
# actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# total_episodes = 10000
# increment = total_episodes//100+1
# # Discount factor for future rewards
# gamma = 0.99
# # Used to update target networks
# tau = 0.005

# buffer = Buffer(500, 64,num_states = 5)

# # To store reward history of each episode
# ep_reward_list = []
# # To store average reward history of last few episodes
# avg_reward_list = []

# # Takes about 4 min to train
# for ep in range(total_episodes):

#     prev_state = env.reset()
#     episodic_reward = 0

#     while True:
#         # Uncomment this to see the Actor in action
#         # But not in a python notebook.
#         env.render()
#         tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

#         action = policy(tf_prev_state, ou_noise)
#         # Recieve state and reward from environment.

#         state, reward, done, info = env.step(action)
#         # print("action : ",action)
#         # print("state, reward, done, info : ", state, reward, done, info)
#         buffer.record((prev_state, action, reward, state))
#         episodic_reward += reward

#         buffer.learn()
#         update_target(target_actor.variables, actor_model.variables, tau)
#         update_target(target_critic.variables, critic_model.variables, tau)

#         # End this episode when `done` is True
#         if done:
#             break

#         prev_state = state

#     ep_reward_list.append(episodic_reward)

#     # Mean of last 4*increment episodes
    
#     avg_reward = np.mean(ep_reward_list[-4*increment:])
#     if ep%increment == 0 : 
#         print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#     avg_reward_list.append(avg_reward)



# # Plotting graph
# # Episodes versus Avg. Rewards
# plt.plot(avg_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Epsiodic Reward")
# plt.show()

# # Save the weights
# actor_model.save_weights("h5f_files/MRs.h5")
# critic_model.save_weights("h5f_files/MRs.h5")

# target_actor.save_weights("h5f_files/MRs.h5")
# target_critic.save_weights("h5f_files/MRs.h5")


# ________________________________________________________________________
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
# actor.add(Activation('softsign'))
actor.add(Activation('tanh'))
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
memory = SequentialMemory(limit=20000, window_length=1)
random_process = None # OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.6, mu=0, sigma=0.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, batch_size=256, nb_steps_warmup_critic=2000, nb_steps_warmup_actor=1000,
                   gamma=.99,random_process=random_process, target_model_update=5e-2) #
agent.compile(Adam(lr=0.001,  clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something!
mode = 'train'
if mode == 'train':
    hist = agent.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=10000)
    filename = '600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'
    # # we save the history of learning, it can further be used to plot reward evolution
    with open('_experiments/history_ddpg__redetorcs'+filename+'.pickle', 'wb') as handle:
         pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # After training is done, we save the final weights.
    agent.save_weights('h5f_files/ddpg_{}_weights.h5f'.format('600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2_action_lim_1'), overwrite=True)

    # # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=100, visualize=True, nb_max_episode_steps=1000)
elif mode == 'test':
    env.set_test_performace() # Define the initialization as performance test
    env.set_save_experice()   # Save the test to plot the results after
    agent.load_weights('h5f_files/ddpg_{}_weights.h5f'.format('600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'))
    agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=1000)
