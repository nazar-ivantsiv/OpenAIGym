import gym
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # To suppress TF warnings
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import os
import numpy as np
env = gym.make('LunarLander-v2')


class DQN:

    """ Implementation of Deep Q-Learning algorithm """

    def __init__(self, env, alpha=0.001, gamma=.99, epsilon=1.0, epsilon_min=.01, epsilon_max=1.0, epsilon_decay=0.996, batch_size=64, seed=0):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay

        self.env = env
        self.env.seed(seed)
        np.random.seed(seed)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        self.max_steps = self.env._max_episode_steps   #1000 in LunarLander
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.asarray([i[0] for i in minibatch])
        actions = np.asarray([i[1] for i in minibatch])
        rewards = np.asarray([i[2] for i in minibatch])
        next_states = np.asarray([i[3] for i in minibatch])
        dones = np.asarray([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.arange(self.batch_size)
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        act_values = np.squeeze(self.model.predict(state))
        return np.argmax(act_values)  # Exploitation

    def train(self, episodes):
        rewards = []
        for e in range(episodes):
            self.epsilon = self.decay_function(e, episodes)
            state = self.env.reset()
            state = state.reshape(1, 8)
            total_reward = 0
            for step in range(self.max_steps):
                action = self.act(state)
                # self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = next_state.reshape(1, 8)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                state = next_state
                if done:
                    print(f'episode: {e}/{episodes}, total_reward: {total_reward}')
                    break
            rewards.append(total_reward)

            # Average score of last 100 episode
            avg_reward = np.mean(rewards[-100:])
            is_solved = avg_reward > 200.0
            if is_solved:
                print('\n Task Completed! \n')
                self.save_weights(f'dqn_e{e}.h5')
                break
            print(f'[{e}/{episodes}] Average over last 100 episodes: {avg_reward:.2f}')
            print(f'epsilon={self.epsilon:.4f}')

            # Save intermediate model weights every N episodes
            if not (e % 50) and e:
                self.save_weights(f'dqn_e{e}.h5')

        return rewards

    def evaluate(self, episodes=1):
        curr_epsilon = self.epsilon
        self.epsilon = 0.0
        for e in range(episodes):
            done = False
            state = self.env.reset()
            state = state.reshape(1, 8)
            score = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                next_state = next_state.reshape(1, 8)
                state = next_state
            print(f'episode: {e}/{episodes}, score: {score}')
        self.epsilon = curr_epsilon

    def decay_function(self, episode, total_episodes):
        return np.clip(1.0 - np.log10((episode + 1) / (total_episodes * 0.1)), self.epsilon_min, self.epsilon_max)

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def save_weights(self, weights_fn='model_weights.h5'):
        folder = 'checkpoints'
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.model.save_weights(os.path.join(folder, weights_fn))
        print(f'Model weights saved to : {os.path.join(folder, weights_fn)}')

    def load_weights(self, weights_fn):
        if weights_fn is not None:
            if os.path.isfile(weights_fn):
                self.model.load_weights(weights_fn)
            else:
                raise ValueError(f'Invalid file name specified: {weights_fn}')


def create_env():
        return gym.make('LunarLander-v2')


if __name__ == '__main__':
    env = create_env()
    
    agent = DQN(env)
    episodes = 600  # Should be enough to converge to optimal policy
    reward = agent.train(episodes)
    np.save(f'dqn_reward_{episodes}.npy', np.asarray(reward))

#     # Resume training
#     agent = DQN(env, lr=0.0001, epsilon=0.1, epsilon_max=0.1)
#     agent.load_weights('checkpoints/dqn_e350.h5')
#     episodes = 800
#     reward = agent.train(episodes)
#     np.save(f'dqn_reward_{episodes}.npy', np.asarray(reward))
    
    # agent.load_weights('dqn_e200.h5')
#     agent.evaluate(5)
