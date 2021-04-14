import gym
import sys
import argparse
import numpy as np


env = gym.make('LunarLander-v2')

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--episodes', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=300)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

score_list = []
for e in range(args.episodes):
    done = False
    score = 0.0
    i = 0
    env.reset()
    while (not done) and (i <= args.max_steps):
        if args.render:
            env.render()
        action = np.random.randint(env.action_space.n)
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = next_state
        i += 1
    score_list.append(score)
    print(f'episode: {e}/{args.episodes}, score: {score}')
print(f'avg_score: {np.average(score_list)}; score_std: {np.std(score_list)}')
env.close()
