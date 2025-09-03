# ch4_dynamic_programming/examples/run_value_iteration.py
import numpy as np
from ch4_dynamic_programming.gridworld import GridWorld4x4
from ch4_dynamic_programming.value_iteration import value_iteration

if __name__ == "__main__":
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3), gamma=1.0)
    V, pi = value_iteration(env)
    s0 = env.s2i[(0, 0)]
    print("V(start):", V[s0])
    print("Greedy action at start:", np.argmax(pi[s0]))
