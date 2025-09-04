# ch5_monte_carlo/tests/test_mc_control.py
import numpy as np

from ch5_monte_carlo.gridworld import GridWorld4x4
from ch5_monte_carlo.examples.mc_control_onpolicy_gridworld import mc_control_onpolicy, ACTIONS
from ch5_monte_carlo.examples.mc_control_es_gridworld import mc_es_control
from ch5_monte_carlo.examples.rollout_utils import rollout_greedy_from_soft, rollout_greedy_es


ARROWS = {0: "â†’", 1: "â†", 2: "â†“", 3: "â†‘"}

def rollout_greedy_es(env: GridWorld4x4, pi, max_steps=64):
    """Roll out deterministic greedy policy 'pi' (state -> action)."""
    s = env.S[np.random.randint(len(env.S))]
    steps = 0
    while s != env.goal and steps < max_steps:
        a = pi[s]
        s_idx = env.s2i[s]
        trans = env.P[s_idx][a]
        probs = [p for (p, _, _) in trans]
        i = np.random.choice(len(trans), p=probs)
        _, sp_idx, _ = trans[i]
        s = env.i2s[sp_idx]
        steps += 1
    return s == env.goal, steps

def greedy_from_soft(pi_soft, s):
    """Pick argmax_a pi(a|s) from dict keyed by (s,a)."""
    return max(ACTIONS, key=lambda a: pi_soft[(s, a)])

def rollout_greedy_from_soft(env: GridWorld4x4, pi_soft, max_steps=64):
    """Roll out using greedy action from Îµ-soft policy probabilities."""
    s = env.S[np.random.randint(len(env.S))]
    steps = 0
    while s != env.goal and steps < max_steps:
        a = greedy_from_soft(pi_soft, s)
        s_idx = env.s2i[s]
        trans = env.P[s_idx][a]
        probs = [p for (p, _, _) in trans]
        i = np.random.choice(len(trans), p=probs)
        _, sp_idx, _ = trans[i]
        s = env.i2s[sp_idx]
        steps += 1
    return s == env.goal, steps

def success_rate(trial_fn, trials=100):
    succ = 0
    total_steps = 0
    for _ in range(trials):
        ok, steps = trial_fn()
        succ += int(ok)
        total_steps += steps
    return succ / trials, total_steps / trials

def test_mc_es_gridworld_reaches_goal():
    np.random.seed(0)
    env = GridWorld4x4(step_reward=0.0, goal=(0, 3))

    # Train MC-ES (keep episodes modest so CI stays fast)
    Q, pi = mc_es_control(env, episodes=1500, gamma=0.9)

    sr, avg_steps = success_rate(lambda: rollout_greedy_es(env, pi), trials=100)

    # Expect high success and reasonable path length
    assert sr >= 0.9, f"MC-ES success rate too low: {sr:.2f}"
    assert avg_steps <= 25, f"MC-ES average steps too high: {avg_steps:.1f}"

def test_mc_onpolicy_gridworld_reaches_goal():
    np.random.seed(1)
    env = GridWorld4x4(step_reward=0.0, goal=(0, 3))

    # Train on-policy MC with Îµ-soft behavior
    Q, pi_soft = mc_control_onpolicy(env, episodes=2000, gamma=0.9, epsilon=0.1)

    sr, avg_steps = success_rate(lambda: rollout_greedy_from_soft(env, pi_soft), trials=100)

    # Expect robust success with a slightly looser bound than MC-ES
    assert sr >= 0.85, f"On-policy MC success rate too low: {sr:.2f}"
    assert avg_steps <= 28, f"On-policy MC average steps too high: {avg_steps:.1f}"

