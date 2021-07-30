import numpy as np 
import time
import random
import gym

env = gym.make("Taxi-v3")

state_space = env.observation_space.n 
action_space = env.action_space.n

# define q table
q_table = np.zeros((state_space, action_space))

number_of_episodes = 10000
number_of_steps_per_episodes = 1000

learning_rate = 0.1
discount_rate = 0.9

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

reward_all_episodes = []

for episode in range(number_of_episodes):
    state = env.reset()
    done = False
    current_episode_reward = 0

    for step in range(number_of_steps_per_episodes):

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        q_table[state, action] = q_table[state, action] * (1-learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        current_episode_reward += reward

        if done == True:
            break

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    reward_all_episodes.append(current_episode_reward)


#calculate the average reward per 1000 episodes
rewards_per_thousand_episodes = np.split(np.array(reward_all_episodes), number_of_episodes/1000)
count = 1000
print("***** avg reward per 1000 episodes *****")
for r in rewards_per_thousand_episodes:
    print(count, " : ", str(sum(r/1000)))
    count += 1000

# print q table
print("\n\n ****** q table ******")
print(q_table)


for episode in range(3):
    state = env.reset()
    done = False
    print("EPISODE ", episode+1, "\n\n")
    time.sleep(1)

    for step in range(number_of_steps_per_episodes):
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        
        if done==True:
            env.render()
            print(reward)
            if reward == 20:
                print("*** Success *** \n\n")
                time.sleep(3)

            else:
                print("*** Wrong drop ***\n\n")
                time.sleep(3)
            break
        
        state = new_state

env.close()






