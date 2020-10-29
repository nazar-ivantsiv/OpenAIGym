# Landing pad is always at coordinates (0,0). Coordinates are the first
# two numbers in state vector. Reward for moving from the top of the screen
# to landing pad and zero speed is about 100..140 points. If lander moves
# away from landing pad it loses reward back. Episode finishes if the lander
# crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
# Solved is 200 points. Landing outside landing pad is possible. Fuel is
# infinite, so an agent can learn to fly and then land on its first attempt.
# Four discrete actions available: do nothing, fire left orientation engine,
# fire main engine, fire right orientation engine.


import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import os
import numpy as np
env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


def decay_function(episode):
    return max(min_epsilon, min(max_epsilon, 1.0 - math.log10((episode + 1) / (total_train_episodes*0.1))))


def set_buckets_and_actions():
    number_of_buckets = (5,5,5,5,5,5,2,2) #buckets in each dimension
    number_of_actions = env.action_space.n
    
    #Creating a 2-tuple with the original bounds of each dimension
    state_value_bounds = list(zip(env.observation_space.low,env.observation_space.high))
    
    #New bound values for each dimension
    state_value_bounds[0] = [-1,1]      #Position x
    state_value_bounds[1] = [-1,1]    #Position y
    state_value_bounds[2] = [-1,1]        #vel x
    state_value_bounds[3] = [-1,1]    #vel y
    state_value_bounds[4] = [-1,1]        #angle
    state_value_bounds[5] = [-1,1]        #angular vel
    state_value_bounds[6] = [0,1]
    state_value_bounds[7] = [0,1]
    
    return number_of_buckets, number_of_actions, state_value_bounds


def bucketize(state):
    bucket_indexes = []
    for i in range(len(state)):
        if state[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_value_bounds[i][1]:
            bucket_index = number_of_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (number_of_buckets[i]-1) * state_value_bounds[i][0]/bound_width
            scaling = (number_of_buckets[i]-1) / bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)


def Generate_episode(epsilon, q_table, max_env_steps):
    # Control variables
    total_reward = 0
    done = False
        
    trayectory = []
        
    # Initialize S
    # Reset the environment getting the initial state
    bucket_state = bucketize(env.reset())

    # Loop for each step of episode:
    for step in range(max_env_steps):
            #print("step ", step)

        # Choose A from S using a soft policy derived from Q (e.g., epsilon-greedy)
        action = choose_action(q_table, bucket_state, epsilon)
            #print(q_table[bucket_state])
            #print("action ", action)

        # Take the action A, observe R, S'
        new_state, reward, done, info = env.step(action)
        bucket_new_state = bucketize(new_state)
            #print("reward ", reward)
            
        trayectory.append([bucket_state, action, reward])
            
        # new_state is now the current state
        bucket_state = bucket_new_state

        total_reward += reward

        # if done, finish the episode
        if done:
            break
    
    return trayectory, total_reward


class QTable:

    """ Implementation of Q-table algorithm """

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.qtable = np.zeros(number_of_buckets + (number_of_actions,))

    def act(self, state):
        """
        Args:
            state -- quantized state representation (result of 'bucketize' function)
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        return np.argmax(self.qtable(state))  # Exploitation

    def update_table(self):
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.arange(self.batch_size)
        targets_full[[ind], [actions]] = targets

    def save_qtable(self, qtable_fn='model_weights.npy'):
        self.qtable.save(qtable_fn)
        print(f'Q-table saved to : {qtable_fn}')

    def load_qtable(self, qtable_fn):
        if qtable_fn is not None:
            if os.path.isfile(qtable_fn):
                self.qtable = np.load(qtable_fn)
            else:
                raise ValueError(f'Invalid file name specified: {qtable_fn}')

    def train(self, env, episodes):
            visits_counter = np.zeros(number_of_buckets + (number_of_actions,))
            rewards = []
            max_steps = env._max_episode_steps   #1000 in LunarLander
            for e in range(episodes):
                epsilon = decay_function(e)
                trajectory, total_reward = Generate_episode(epsilon, q_table, max_steps)

                # Update the Q-table values starting from the terminal state
                G = 0  # Init Return value
                for t in reversed(range(len(trajectory))):
                    s_t, a_t, r_t = trajectory[t]
                    G = gamma*G + r_t
                    if not [s_t, a_t] in [[x[0], x[1]] for x in trajectory[0:t]]:
                        visits_counter[s_t][a_t] += 1
                        self.q_table[s_t][a_t] += (G - self.q_table[s_t][a_t]) / visits_counter[s_t][a_t]
                rewards.append(total_reward)

                print("episode: {}/{}, score: {}".format(e, episodes, score)) 

                # Average score of last 100 episode
                avg_reward = np.mean(rewards[-100:])
                is_solved = avg_reward > 200.0
                if is_solved:
                    print('\n Task Completed! \n')
                    self.save_weights(f'dqn_e{e}.h5')
                    break
                print("Average over last 100 episode: {0:.2f} \n".format(avg_reward))

                # Save intermediate q-table every N episodes
                if not (e % 50) and e:
                    self.save_qtable(f'qtable_e{e}.npy')

            return rewards

    def evaluate(self, env, episodes=1):



                #env = create_env()
        total_test_episodes = 10
        q_tables = np.load('MC_tables.npy')
        q_table = q_tables[0]
        rewards = []
        max_steps = env._max_episode_steps
        number_of_buckets, number_of_actions, state_value_bounds = set_buckets_and_actions()

        # ******* Loop for each episode:
        for episode in range(total_test_episodes):
            #print("***Episode*** ", episode)
            
            # Control variables
            total_rewards = 0
            done =  False
            
            # ******* Initialize S
            # Reset the environment getting the initial state
            bucket_state = bucketize(env.reset())
            
            # *******Loop for each step of episode:
            for step in range(max_steps):
                env.render()
                
                #******* Choose A from S using policy derived from Q (greedy in this case)
                action = np.argmax(q_table[bucket_state])
                #print(action)
                
                # ******* Take the action A, observe R, S'
                new_state, reward, done, info = env.step(action)
                bucket_new_state = bucketize(new_state)
                
                # new_state is now the current state
                bucket_state =  bucket_new_state
                
                total_rewards += reward
                
                if done:
                    rewards.append(total_rewards)
                    print("Score ", total_rewards)
                    break
                        
        env.close()
        print("\nAverage score " + str(sum(rewards)/total_test_episodes))



        for e in range(episodes):
            state = env.reset()
            state = state.reshape(1, 8)
            score = 0
            while not done:
                env.render()
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                next_state = next_state.reshape(1, 8)
                state = next_state
            print("episode: {}/{}, score: {}".format(e, episodes, score))


if __name__ == '__main__':
    # WEIGHTS = r'dqn_e50.h5'

    print(env.observation_space)
    print(env.action_space)

    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    episodes = 400
    loss = agent.train(env, episodes)
    plt.plot(np.arange(1, len(loss)+1, 2), loss[::2])
    plt.show()

    agent.evaluate(env)


# TODO:
#   - Add comments for important steps. Match with theory, decks.
#   - Replace reshape() with flat array indexing
#   - Standardize all 'format' strings
