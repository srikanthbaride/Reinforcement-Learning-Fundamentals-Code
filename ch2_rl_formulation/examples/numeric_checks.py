import numpy as np
from ..gridworld import GridWorld4x4
from ..evaluation import policy_evaluation, q_from_v, greedy_from_q

def discounted_return_example():
    r = [1, 2, 3]; gamma = 0.9
    return r[0] + gamma*r[1] + (gamma**2)*r[2]

def state_value_example():
    gamma = 0.9
    return -1 + gamma*(-1) + (gamma**2)*(-1) + (gamma**3)*10

def gridworld_vq_under_fixed_policy(gamma: float = 1.0):
    env = GridWorld4x4(step_reward=-1.0, goal=(0, 3))
    pi = np.zeros(env.num_states, dtype=int)
    for s in range(env.num_states):
        i, j = env.i2s[s]
        if s == env.terminal: 
            pi[s] = 0; continue
        pi[s] = 0 if j < 3 else 3
    V = policy_evaluation(env, pi, gamma=gamma, theta=1e-10)
    Q = q_from_v(env, V, gamma=gamma)
    pi_greedy = greedy_from_q(Q)
    return V.reshape(4,4), Q, pi_greedy.reshape(4,4)

if __name__ == "__main__":
    print("G0 example (should be 5.23):", round(discounted_return_example(), 2))
    print("v_pi example (should be 4.58):", round(state_value_example(), 2))
    V, Q, pi_g = gridworld_vq_under_fixed_policy(gamma=1.0)
    print("\nGridworld V under greedy-to-goal policy (gamma=1):")
    print(np.array_str(np.round(V, 0), precision=0))
    print("\nGreedy-from-Q policy indices (0:R,1:L,2:D,3:U):")
    print(pi_g)
