import sys
import argparse
from lunar_lander_dqn import DQN, create_env

env = create_env()
agent = DQN(env, seed=42)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-e', '--episodes', type=int, default=1)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

if args.checkpoint:
    agent.load_weights(args.checkpoint)
agent.evaluate(episodes=args.episodes, render=args.render)
