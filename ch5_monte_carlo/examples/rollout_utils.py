import numpy as np

def greedy_from_soft(pi_soft):
    return np.argmax(pi_soft, axis=1)

def rollout_greedy_from_soft(env, pi_soft, max_steps=200):
    greedy = greedy_from_soft(pi_soft)
    s = env.reset()
    steps = 0
    for _ in range(max_steps):
        a = greedy[env.s2i[s]]
        sp, r, done = env.step(a)
        steps += 1
        s = sp
        if done:
            return True, steps
    return False, steps

def rollout_greedy_es(env, Q, max_steps=200):
    s = env.reset()
    steps = 0
    for _ in range(max_steps):
        a = np.argmax(Q[env.s2i[s]])
        sp, r, done = env.step(a)
        steps += 1
        s = sp
        if done:
            return True, steps
    return False, steps
