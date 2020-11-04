import sys
from lunar_lander_qtable import QTable, create_env

env = create_env()
agent = QTable(env, seed=123)
if len(sys.argv) < 3:
    agent.evaluate(int(sys.argv[1]))
else:
    agent.load_qtable(sys.argv[1])
    agent.evaluate(int(sys.argv[2]))
