

# https://medium.com/@gabogarza/deep-reinforcement-learning-policy-gradients-8f6df70404e6


import gym
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # To suppress TF warnings
from collections import deque
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

import os
import numpy as np


class PG:

    """ Implementation of Policy Gradient algorithm """

    def __init__(self, env, alpha=0.01, gamma=.95, epsilon=1.0, epsilon_min=.01, epsilon_max=1.0, epsilon_decay=0.996, batch_size=64, seed=0):
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
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def remember(self, state, action, reward, next_state, done):
        action_arr = np.zeros(self.action_space)
        action_arr[action] = 1.0
        self.memory.append((state, action_arr, reward))

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
        # Reshape observation to (num_features, 1)
        observation = observation[:, np.newaxis]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

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



        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: np.vstack(self.episode_observations).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        return discounted_episode_rewards_norm



        return rewards

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def evaluate(self, episodes=1, render=False):
        curr_epsilon = self.epsilon
        self.epsilon = 0.0
        score_list = []
        for e in range(episodes):
            done = False
            state = self.env.reset()
            state = state.reshape(1, 8)
            score = 0
            while not done:
                if render:
                    self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                next_state = next_state.reshape(1, 8)
                state = next_state
            score_list.append(score)
            print(f'episode: {e}/{episodes}, score: {score}')
        print(f'avg_score: {np.average(score_list)}; score_std: {np.std(score_list)}')
        self.epsilon = curr_epsilon

    def decay_function(self, episode, total_episodes):
        return np.clip(1.0 - np.log10((episode + 1) / (total_episodes * 0.1)), self.epsilon_min, self.epsilon_max)

    def build_model(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 10
        units_layer_2 = 10
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def save_weights(self, weights_fn='model_weights.h5'):
        folder = 'checkpoints'
        if not os.path.exists(folder):
            os.mkdir(folder)
        save_path = self.saver.save(self.sess, weights_fn)
        print(f'Model weights saved to : {os.path.join(folder, weights_fn)}')

    def load_weights(self, weights_fn):
        if weights_fn is not None:
            if os.path.isfile(weights_fn):
                self.load_path = load_path
                self.saver.restore(self.sess, self.load_path)
            else:
                raise ValueError(f'Invalid file name specified: {weights_fn}')


def create_env():
        return gym.make('LunarLander-v2')


if __name__ == '__main__':
    env = create_env()
    
    agent = DQN(env)
    episodes = 600  # Should be enough to satisfy env criterion (reward 200+)
    reward = agent.train(episodes)
    np.save(f'dqn_reward_{episodes}.npy', np.asarray(reward))

    # # Resume training
    # agent = DQN(env, lr=0.0001, epsilon=0.1, epsilon_max=0.1)
    # agent.load_weights('checkpoints/dqn_e350.h5')
    # episodes = 800
    # reward = agent.train(episodes)
    # np.save(f'dqn_reward_{episodes}.npy', np.asarray(reward))
    
    # agent.load_weights('dqn_e200.h5')
    # agent.evaluate(5)
