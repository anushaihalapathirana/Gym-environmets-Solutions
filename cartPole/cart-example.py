import gym
import time
import numpy as np
import math

env = gym.make("CartPole-v1")

action_space = env.action_space.n
state_space = env.observation_space.shape[0]

number_of_episodes = 2000
# number_of_steps_per_episodes = 100

learning_rate = 0.1
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.99995

# discrete_state_size = [20] * state_space
discrete_state_size =[30, 30, 50, 50]

# discrete_state_window_size = (high-low) / discrete_state_size
discrete_state_window_size = np.array([0.25, 0.25, 0.01, 0.1])

# initialize q table
q_table = np.random.uniform(low=0, high=1, size = (discrete_state_size + [env.action_space.n])) 

reward_all_episodes = []

def get_discrete_state(state):
    # discrete_state = (state - env.observation_space.low) / discrete_state_window_size
    discrete_state =  state/discrete_state_window_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

for episode in range(number_of_episodes):
    state = get_discrete_state(env.reset())

    done = False
    current_episode_reward = 0

    while not done:
        exploration_rate_threshold = np.random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        # update q table
        if not done: 
            max_future_q = np.max(q_table[new_discrete_state])

            current_q = q_table[state + (action,)]

            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_rate * max_future_q)

            q_table[state + (action,)] = new_q

        state = new_discrete_state
        current_episode_reward += reward

        if done == True:
            break
    
    # if exploration_rate > 0.05: #epsilon modification
    #     if current_episode_reward > reward and episode > 10000:
    #         exploration_rate = math.pow(exploration_decay_rate, episode - 10000)
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    reward_all_episodes.append(current_episode_reward)
        
env.close()

#calculate the average reward per 1000 episodes
# rewards_per_thousand_episodes = np.split(np.array(reward_all_episodes), number_of_episodes/10000)
# count = 10000
# print("***** avg reward per 1000 episodes *****")
# for r in rewards_per_thousand_episodes:
#     print(count, " : ", str(sum(r/10000)))
#     count += 10000

for episode in range(3):
    state = get_discrete_state(env.reset())
    done = False
    print("EPISODE ", episode+1, "\n\n")
    time.sleep(1)

    while not done:
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state])
        new_state, reward, done, info = env.step(action)
        
        if done==True:
            env.render()
            print(reward)
            time.sleep(0.5)
            break
        
        state = get_discrete_state(new_state)

env.close()






