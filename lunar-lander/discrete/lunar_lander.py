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


class DQN:

    """ Implementation of Deep Q-Learning algorithm """

    def __init__(self, env):
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        # self.epsilon_decay = .996

        self.env = env
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        self.max_steps = self.env._max_episode_steps   #1000 in LunarLander
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def remember(self, state, action, reward, next_state, done):
        """  """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """  """
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

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.arange(self.batch_size)
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Exploitation

    def train(self, episodes):
        rewards = []
        for e in range(episodes):
            self.epsilon = self.decay_function(e, episodes)

            print(f'epsilon: {self.epsilon}')

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
                    print("episode: {}/{}, total_reward: {}".format(e, episodes, total_reward))
                    break
            rewards.append(total_reward)

            # Average score of last 100 episode
            avg_reward = np.mean(rewards[-100:])
            is_solved = avg_reward > 200.0
            if is_solved:
                print('\n Task Completed! \n')
                self.save_weights(f'dqn_e{e}.h5')
                break
            print("Average over last 100 episode: {0:.2f} \n".format(avg_reward))

            # Save intermediate model weights every N episodes
            if not (e % 50) and e:
                self.save_weights(f'dqn_e{e}.h5')

        return rewards

    def evaluate(self, episodes=1):
        done = False
        for e in range(episodes):
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
            print("episode: {}/{}, score: {}".format(e, episodes, score))

    def decay_function(self, episode, total_episodes):
        return max(self.epsilon_min, min(1.0, 1.0 - np.log10((episode + 1) / (total_episodes * 0.1))))

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def save_weights(self, weights_fn='model_weights.h5'):
        self.model.save_weights(weights_fn)
        print(f'Model weights saved to : {weights_fn}')

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

    # agent = DQN(env)
    # episodes = 400
    agent = DQN(env)
    agent.load_weights('dqn_e150.h5')
    episodes = 250
    loss = agent.train(episodes)
    plt.plot(np.arange(1, len(loss)+1, 2), loss[::2])
    plt.show()

    agent.evaluate()


# TODO:
#   - Add comments for important steps. Match with theory, decks.
#   - Replace reshape() with flat array indexing
