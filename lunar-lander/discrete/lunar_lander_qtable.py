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


class QTable:

    """ Implementation of Q-Table algorithm """

    def __init__(self, env):
        self.epsilon = 1.0
        self.gamma = .99
        self.epsilon_min = .01
        self.epsilon_max = 1.0
        self.lr = 0.001
        # self.epsilon_decay = .996

        self.env = env
        self.env.seed(0)
        np.random.seed(0)
        self.action_space = self.env.action_space.n
        self.max_steps = self.env._max_episode_steps   #1000 in LunarLander
        n_buckets, n_actions, state_bounds = self._init_buckets_and_actions()
        self.n_buckets = n_buckets
        self.n_actions = n_actions
        self.state_bounds = state_bounds
        self.q_table = np.zeros(n_buckets + (n_actions,))
        self.visits_counter = np.zeros(n_buckets + (n_actions,))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def update_table(self, trajectory):
        """ Update the Q-table values starting from the terminal state """
        g = 0.0
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            g = self.gamma * g + r_t
            if not [s_t, a_t] in [[x[0], x[1]] for x in trajectory[0:t]]:
                self.visits_counter[s_t][a_t] += 1
                self.q_table[s_t][a_t] += (g - self.q_table[s_t][a_t]) / self.visits_counter[s_t][a_t]

    def train(self, episodes):
            rewards = []
            for e in range(episodes):
                trajectory = []
                state = self._bucketize(self.env.reset())
                total_reward = 0
                for step in range(self.max_steps):
                    # self.env.render()
                    action = self.act(state)
                    new_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    new_state = self._bucketize(new_state)
                    trajectory.append([state, action, reward])
                    state = new_state
                    if done:
                        print("episode: {}/{}, total_reward: {}".format(e, episodes, total_reward))
                        break
                self.update_table(trajectory)
                rewards.append(total_reward)

                # Average score of last 100 episode
                avg_reward = np.mean(rewards[-100:])
                is_solved = avg_reward > 200.0
                if is_solved:
                    print('\n Task Completed! \n')
                    self.save_qtable(f'qtbl_e{e}.h5')
                    break
                print("Average over last 100 episode: {0:.2f} \n".format(avg_reward))

                # Save intermediate q-table every N episodes
                if not (e % 50) and e:
                    self.save_qtable(f'qtbl_e{e}.npy')

            return rewards

    def evaluate(self, episodes=1):
        curr_epsilon = self.epsilon
        self.epsilon = 0.0
        for e in range(episodes):
            done = False
            state = self._bucketize(self.env.reset())
            score = 0.0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self._bucketize(next_state)
                score += reward
                state = next_state
            print("episode: {}/{}, score: {}".format(e, episodes, score))
        self.epsilon = curr_epsilon

    def decay_function(self, episode, total_episodes):
        return np.clip(1.0 - np.log10((episode + 1) / (total_episodes * 0.1)), self.epsilon_min, self.epsilon_max)

    def save_qtable(self, q_table_fn='qtbl.npy'):
        np.save(q_table_fn, self.q_table)
        print(f'Q-table saved to : {q_table_fn}')

    def load_qtable(self, q_table_fn):
        if q_table_fn is not None:
            if os.path.isfile(q_table_fn):
                self.q_table = np.load(q_table_fn)
                if len(self.q_table.shape) == 10:
                    self.q_table = self.q_table[0]
            else:
                raise ValueError(f'Invalid file name specified: {q_table_fn}')

    def _init_buckets_and_actions(self):
        n_buckets = (5, 5, 5, 5, 5, 5, 2, 2) # buckets in each dimension
        n_actions = self.env.action_space.n
        
        # Creating a 2-tuple with the original bounds of each dimension
        state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        
        # New bound values for each dimension
        state_bounds[0] = [-1,1]      # position x
        state_bounds[1] = [-1,1]      # position y
        state_bounds[2] = [-1,1]      # vel x
        state_bounds[3] = [-1,1]      # vel y
        state_bounds[4] = [-1,1]      # angle
        state_bounds[5] = [-1,1]      # angular vel
        state_bounds[6] = [0,1]
        state_bounds[7] = [0,1]
        
        return n_buckets, n_actions, state_bounds

    def _bucketize(self, state):
        # TODO: Refactor for performance
        bucket_indexes = []
        for i in range(len(state)):
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.n_buckets[i] - 1
            else:
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.n_buckets[i]-1) * self.state_bounds[i][0]/bound_width
                scaling = (self.n_buckets[i]-1) / bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indexes.append(bucket_index)
            
        return tuple(bucket_indexes)


def create_env():
    return gym.make('LunarLander-v2')


if __name__ == '__main__':
    env = create_env()

    agent = QTable(env)
    # episodes = 10000
    # loss = agent.train(episodes)
    # plt.plot(np.arange(1, len(loss)+1, 2), loss[::2])
    # plt.show()

    agent.load_qtable('qtbl_e10000.npy')
    agent.evaluate(5)


# TODO:
#   - Add comments for important steps. Match with theory, decks.
#   - Standardize all 'format' strings
