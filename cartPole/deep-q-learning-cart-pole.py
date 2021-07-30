import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v1')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

#Create a deep learning model with keras
def build_model(state_space, action_space):
    model = Sequential()
    model.add(Flatten(input_shape = (1, state_space)))
    model.add(Dense(24, activation = 'relu' ))
    model.add(Dense(24, activation = 'relu' ))
    model.add(Dense(action_space, activation = 'linear' ))
    return model

# Build the agent with Keras-RL
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length=1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy, nb_actions = actions, nb_steps_warmup = 10, target_model_update = 1e-2)
    return dqn


#create the model
model = build_model(state_space, action_space)

# create agent 
dqn = build_agent(model, action_space)
dqn.compile(Adam(lr=1e-3), metrics = ['mae'])
dqn.fit(env, nb_steps = 10000, visualize = False ,verbose = 1)

# print(model.summary())

# test the trained model
scores = dqn.test(env, nb_episodes = 100, visualize = False)
print(np.mean(scores.history['episode_reward']))

# visualize
_ = dqn.test(env, nb_episodes=10, visualize = True)