import sys
import argparse
from lunar_lander_qtable_td import QTable, create_env

env = create_env()
agent = QTable(env, seed=42)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-e', '--episodes', type=int, default=1)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

if args.checkpoint:
    agent.load_qtable(args.checkpoint)
agent.evaluate(episodes=args.episodes, render=args.render)
