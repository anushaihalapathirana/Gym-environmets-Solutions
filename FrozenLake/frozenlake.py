import gym
import time
import numpy as np
import random

env = gym.make("FrozenLake-v0")

state_space = env.observation_space.n
action_space = env.action_space.n

#initalize the q table. rows = number of states and cols = number of actions
q_table = np.zeros((state_space, action_space))

number_of_episodes = 10000
steps_per_episode = 1000

learning_rate = 0.1 # alpha
discount_rate = 0.99 # gamma

# related to the epsilon greedy strategy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q learning algorithm
for episode in range(number_of_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(steps_per_episode):

        #exploration trade off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        # update q table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1-learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        # env.render()

        if done == True:
            break

    # once the episode is over we need to update exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

# env.close()
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


# agent play frozen lake using trained data

for episode in range(3):
    state = env.reset()
    done = False
    print("EPISODE ", episode+1, "\n\n")
    time.sleep(1)

    for step in range(steps_per_episode):
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done==True:
            env.render()
            if reward == 1:
                print("*** Reach the goal *** \n\n")
                time.sleep(3)

            else:
                print("*** you fell in a hole ***\n\n")
                time.sleep(3)
            break
        
        state = new_state

env.close()







