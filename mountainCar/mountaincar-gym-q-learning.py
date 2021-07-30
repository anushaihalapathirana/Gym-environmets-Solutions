import gym
import numpy as np
import matplotlib.pyplot as plt

# this env has 3 actions. push car right, do nothing, push car left.
env = gym.make("MountainCar-v0")
# env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
# print(env.action_space)

LEARNING_RATE = 0.1
DISCOUNT = 0.95 #  weight
EPISODES  = 20

SHOW_EVERY = 5

DISCRETE_OBSERVATION_SIZE = [20] * len(env.observation_space.high)

print(DISCRETE_OBSERVATION_SIZE)

discrete_observation_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

print(DISCRETE_OBSERVATION_SIZE)

# creating the q table
q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OBSERVATION_SIZE + [env.action_space.n])) # 20 X 20 X # of actions size 3D q table 

episode_rewards = []
# aggregate_episode_reward_dictionary = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_observation_window_size
    return tuple(discrete_state.astype(np.int))


for episodes in range(EPISODES):

    episode_reward = 0
    if episodes % SHOW_EVERY == 0:
        print("Training")
    
    discrete_state = get_discrete_state(env.reset())

    done = False

    # state is 2 values- position and velocity of the car
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
            
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
            print(f"We made it in {episodes}")
        
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episodes >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    episode_rewards.append(episode_reward)


    if not episodes % SHOW_EVERY:
        np.save(f"qtables/{episodes}-qtable.npy", q_table)
        # average_reward = sum(episode_rewards[-SHOW_EVERY:])/len(episode_rewards[-SHOW_EVERY:])
        # aggregate_episode_reward_dictionary['ep'].append(episodes)
        # aggregate_episode_reward_dictionary['avg'].append(average_reward)
        # aggregate_episode_reward_dictionary['min'].append(min(episode_rewards[-SHOW_EVERY:]))
        # aggregate_episode_reward_dictionary['max'].append(max(episode_rewards[-SHOW_EVERY:]))

        # print(f"episode : {episodes} avg: {average_reward} min: {min(episode_rewards[-SHOW_EVERY:])}, max: {max(episode_rewards[-SHOW_EVERY:])}")

        
env.close()
