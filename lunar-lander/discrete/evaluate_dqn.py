import sys
from lunar_lander_dqn import DQN, create_env

env = create_env()
agent = DQN(env, seed=42)
if len(sys.argv) < 3:
    agent.evaluate(int(sys.argv[1]))
else:
    agent.load_weights(sys.argv[1])
    agent.evaluate(int(sys.argv[2]))
