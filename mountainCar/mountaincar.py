import numpy as np 
import time
import gym
import random

env = gym.make("MountainCar-v0")

action_space = env.action_space.n 
state1 = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
state2 = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

q_table = np.zeros((len(state1),len(state2), action_space))

number_of_episodes = 5000
number_of_steps_per_episodes = 100

learning_rate = 0.1
discount_rate = 0.9

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

reward_all_episodes = []

def get_discrete_state(state):
    DISCRETE_OBSERVATION_SIZE = [20] * len(env.observation_space.high)
    discrete_observation_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SIZE
    discrete_state = (state - env.observation_space.low) / discrete_observation_window_size
    return tuple(discrete_state.astype(np.int))



for episode in range(number_of_episodes):
    discrete_state = get_discrete_state(env.reset())
    done = False
    current_episode_reward = 0

    while not done:

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        new_discrete_state = get_discrete_state(new_state)

        q_table[discrete_state + (action, )] = q_table[discrete_state + (action, )] * (1-learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_discrete_state]))

        discrete_state = new_discrete_state 
        current_episode_reward += reward

        if done == True:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    reward_all_episodes.append(current_episode_reward)

# print q table
print("\n\n ****** q table ******")
print(q_table)


for episode in range(10):
    state = get_discrete_state(env.reset())
    done = False
    print("EPISODE ", episode+1, "\n\n")
    time.sleep(1)

    while not done:
        env.render()
        # time.sleep(0.3)

        action = np.argmax(q_table[state])
        new_state, reward, done, info = env.step(action)
        
        if done==True:
            env.render()
            if new_state[0] >= env.goal_position:
                print("We reach the goal")
            else:
                print("Sorry ")
            break
        
        state = get_discrete_state(new_state)

env.close()







