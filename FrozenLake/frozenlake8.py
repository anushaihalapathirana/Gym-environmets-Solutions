import numpy as np
import gym
import time
import random

env = gym.make("FrozenLake8x8-v0")

action_space = env.action_space.n
state_space = env.observation_space.n

#create initial q table
q_table = np.zeros((state_space, action_space))

number_of_episodes = 105000
number_of_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.9

exploration_rate = 1
min_exploration_rate = 0.001
max_exploration_rate = 1
exploration_decay_rate = 0.0001

rewards_all_episodes = []

for episode in range(number_of_episodes):
    state = env.reset()

    current_episode_reward = 0
    done = False

    for step in range(number_of_steps_per_episode):

        #select action
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        # update q table
        q_table[state, action] = q_table[state, action] * (1-learning_rate) + learning_rate*(reward + discount_rate*np.max(q_table[new_state,:]))

        state = new_state
        current_episode_reward += reward

        if done:
            break
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(current_episode_reward)

env.close()

#calculate the average reward per 1000 episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), number_of_episodes/1000)
count = 1000
print("***** avg reward per 1000 episodes *****")
for r in rewards_per_thousand_episodes:
    print(count, " : ", str(sum(r/1000)))
    count += 1000

# print q table
print("\n\n ****** q table ******")
print(q_table)




# test
for episode in range(3):
    state = env.reset()
    done = False
    print("EPISODE ", episode+1, "\n\n")
    time.sleep(1)

    for step in range(number_of_steps_per_episode):
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        state = new_state

        if done == True:
            env.render()
            if reward == 1:
                print("*** Reach the goal *** \n\n")
                time.sleep(3)

            else:
                print("*** you fell in a hole ***\n\n")
                time.sleep(3)
            break
env.close()