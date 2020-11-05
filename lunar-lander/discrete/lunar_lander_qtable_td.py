import gym
import random
import matplotlib.pyplot as plt

import os
import numpy as np


class QTable:

    """ Implementation of tabular Q-learning algorithm. Temporal Differenct update. """

    def __init__(self, env, alpha=0.1, gamma=.99, epsilon=1.0, epsilon_min=.01, epsilon_max=1.0, epsilon_decay=0.996, seed=0):
        self.epsilon = epsilon
        self.alpha = alpha  # Learning rate (TD only)
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        
        self.env = env
        self.env.seed(seed)
        np.random.seed(seed)
        self.action_space = self.env.action_space.n
        self.max_steps = self.env._max_episode_steps   #1000 in LunarLander
        n_buckets, n_actions, state_bounds = self._init_buckets_and_actions()
        self.n_buckets = n_buckets
        self.n_actions = n_actions
        self.state_bounds = state_bounds
        self.q_table = np.zeros(n_buckets + (n_actions,))
        self.visits_counter = np.zeros(n_buckets + (n_actions,))

    def act(self, state):
        """ e-greedy policy """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def update_td(self, state, action, reward, new_state):
        """ TD(0) Temporal Difference update for a single transition (step) """
        old_value = self.q_table[state][action]
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))
        self.q_table[state][action] = new_value

    def train(self, episodes):
            rewards = []
            for e in range(episodes):
                self.epsilon = self.decay_function(e, episodes)
#                 if self.epsilon > self.epsilon_min:
#                     self.epsilon *= self.epsilon_decay
                state = self._bucketize(self.env.reset())
                total_reward = 0
                for step in range(self.max_steps):
                    action = self.act(state)
                    new_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    new_state = self._bucketize(new_state)
                    self.update_td(state, action, reward, new_state)
                    state = new_state
                    if done:
#                         print(f'episode: {e}/{episodes}, total_reward: {total_reward}')
                        break
                rewards.append(total_reward)

                # Average score of last 100 episode
                avg_reward = np.mean(rewards[-100:])
                is_solved = avg_reward > 200.0
                if is_solved:
                    print('\n Task Completed! \n')
                    self.save_qtable(f'qtbl_e{e}.h5')
                    break
                if not e % 500:
                    print(f'[{e}/{episodes}] Average over last 100 episodes: {avg_reward:.2f}')
                    print(f'epsilon={self.epsilon:.4f}')

                # Save intermediate q-table every N episodes
                if not (e % 1000) and e:
                    self.save_qtable(f'qtbl_td_e{e}.npy')
            self.env.close()
            
            return rewards

    def evaluate(self, episodes=1):
#         self.env.seed(np.random.randint(1000))
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
            print(f'episode: {e}/{episodes}, score: {score}')
        self.epsilon = curr_epsilon
        self.env.close()

    def decay_function(self, episode, total_episodes):
        return np.clip(1.0 - np.log10((episode + 1) / (total_episodes * 0.1)), self.epsilon_min, self.epsilon_max)

    def save_qtable(self, q_table_fn='qtbl_td.npy'):
        folder = 'checkpoints'
        if not os.path.exists(folder):
            os.mkdir(folder)
        np.save(os.path.join(folder, q_table_fn), self.q_table)
        print(f'Q-table saved to : {os.path.join(folder, q_table_fn)}')

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
        state_bounds[0] = [-1,1]  # x coordinate
        state_bounds[1] = [-1,1]  # y coordinate
        state_bounds[2] = [-1,1]  # x speed
        state_bounds[3] = [-1,1]  # y speed
        state_bounds[4] = [-1,1]  # angle
        state_bounds[5] = [-1,1]  # angular speed
        state_bounds[6] = [0,1]   # if first leg has contact
        state_bounds[7] = [0,1]   # if second leg has contact
        
        return n_buckets, n_actions, state_bounds

    def _bucketize(self, state):
        bucket_indexes = []
        for i in range(len(state)):
            n = self.n_buckets[i] - 1
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = n
            else:
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = n * self.state_bounds[i][0] / bound_width
                scaling = n / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indexes.append(bucket_index)
            
        return tuple(bucket_indexes)

    @property
    def qtbl_2d(self):
        """ Transform original Q-table into 2D [n_states, n_actions] """
        n_actions = self.n_actions
        n_states = np.prod(self.n_buckets)
        qtbl_2d = self.q_table.copy().reshape(n_states, n_actions)
        
        return qtbl_2d

def create_env():
    return gym.make('LunarLander-v2')


if __name__ == '__main__':
    env = create_env()
    agent = QTable(env)
    episodes = 20000
    reward = agent.train(episodes)
    np.save(f'qtbl_td_reward_{episodes}.npy', np.asarray(reward))

#     agent.load_qtable('qtbl_e10000.npy')
#     agent.evaluate(5)
