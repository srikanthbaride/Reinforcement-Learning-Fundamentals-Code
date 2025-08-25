# demo_random_mdp.py
# Run Chapter 2 code on a randomly generated finite MDP.
# - Tries to import your existing functions:
#     ch2_rl_formulation.policy_evaluation.policy_evaluation
#     ch2_rl_formulation.value_iteration.value_iteration
# - If missing, uses light fallback implementations.

from __future__ import annotations
import numpy as np

# -----------------------------
# Try to use your implementations
# -----------------------------
policy_eval_fn = None
value_iter_fn = None

try:
    from ch2_rl_formulation.policy_evaluation import policy_evaluation as _pe
    policy_eval_fn = _pe
except Exception:
    pass

try:
    from ch2_rl_formulation.value_iteration import value_iteration as _vi
    value_iter_fn = _vi
except Exception:
    pass

# -----------------------------
# Fallbacks (only used if import fails)
# -----------------------------
def _fallback_policy_evaluation(P, R, gamma, policy, tol=1e-8, max_iter=10_000):
    """
    P: (S,A,S), R: (S,A,S), policy: (S,A) action-probabilities
    Returns V (S,)
    """
    S, A, _ = P.shape
    V = np.zeros(S, dtype=float)
    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        for s in range(S):
            for a in range(A):
                V_new[s] += policy[s, a] * np.sum(P[s, a] * (R[s, a] + gamma * V))
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V

def _fallback_value_iteration(P, R, gamma, tol=1e-8, max_iter=10_000):
    """
    Standard VI returning (V*, pi*).
    pi* returned as greedy deterministic actions (S,) over A.
    """
    S, A, _ = P.shape
    V = np.zeros(S, dtype=float)
    for _ in range(max_iter):
        V_new = np.empty_like(V)
        for s in range(S):
            q_sa = np.empty(A, dtype=float)
            for a in range(A):
                q_sa[a] = np.sum(P[s, a] * (R[s, a] + gamma * V))
            V_new[s] = np.max(q_sa)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    # Greedy policy
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        q_sa = np.array([np.sum(P[s, a] * (R[s, a] + gamma * V)) for a in range(A)])
        pi[s] = int(np.argmax(q_sa))
    return V, pi

# -----------------------------
# Random MDP generator + helpers
# -----------------------------
def random_mdp(num_states=5, num_actions=3, reward_low=-1.0, reward_high=1.0, seed=None):
    rng = np.random.default_rng(seed)
    # Transition probabilities
    P = rng.random((num_states, num_actions, num_states))
    P /= P.sum(axis=2, keepdims=True)
    # Rewards for (s,a,s')
    R = rng.uniform(reward_low, reward_high, size=(num_states, num_actions, num_states))
    return P, R

def random_uniform_policy(num_states, num_actions, seed=None):
    rng = np.random.default_rng(seed)
    logits = rng.random((num_states, num_actions))
    policy = logits / logits.sum(axis=1, keepdims=True)
    return policy

def simulate_episode(P, R, policy_or_actions, start_state=None, horizon=15, seed=None):
    """
    policy_or_actions: either (S,A) probs or a callable state->action int.
    Returns (trajectory list, total_reward)
    """
    rng = np.random.default_rng(seed)
    S = P.shape[0]
    s = rng.integers(S) if start_state is None else int(start_state)
    traj = []
    total = 0.0
    for t in range(horizon):
        if callable(policy_or_actions):
            a = int(policy_or_actions(s))
        else:
            a = int(rng.choice(P.shape[1], p=policy_or_actions[s]))
        s_next = int(rng.choice(S, p=P[s, a]))
        r = float(R[s, a, s_next])
        traj.append((s, a, r, s_next))
        total += r
        s = s_next
    return traj, total

# -----------------------------
# Main demo
# -----------------------------
def run_demo():
    # Hyperparams
    S, A = 6, 3
    gamma = 0.95
    seed = 42

    print("🔧 Building a random finite MDP …")
    P, R = random_mdp(num_states=S, num_actions=A, seed=seed)
    random_pi = random_uniform_policy(S, A, seed=seed)

    # Policy Evaluation
    print("\n📘 Policy Evaluation under a random stochastic policy:")
    pe_fn = policy_eval_fn or _fallback_policy_evaluation
    V_pi = pe_fn(P, R, gamma, random_pi)
    np.set_printoptions(precision=4, suppress=True)
    print("V^π (first 10 shown):", V_pi[:min(10, V_pi.shape[0])])

    # Value Iteration
    print("\n🏁 Value Iteration (optimal values and a greedy policy):")
    vi_fn = value_iter_fn or _fallback_value_iteration
    # If user's VI returns only V, adapt; otherwise expect (V, pi)
    vi_out = vi_fn(P, R, gamma)
    if isinstance(vi_out, tuple) and len(vi_out) == 2:
        V_star, pi_star = vi_out
    else:
        V_star, pi_star = vi_out, np.array([
            np.argmax([np.sum(P[s, a] * (R[s, a] + gamma * vi_out)) for a in range(A)])
            for s in range(S)
        ])
    print("V* (first 10 shown):", V_star[:min(10, V_star.shape[0])])
    print("π* (deterministic actions):", pi_star)

    # Rollouts
    print("\n🎬 Simulating short rollouts:")
    traj_rand, tot_rand = simulate_episode(P, R, random_pi, horizon=10, seed=seed)
    traj_opt, tot_opt = simulate_episode(P, R, lambda s: pi_star[s], horizon=10, seed=seed+1)
    for i, (label, traj, tot) in enumerate([("random π", traj_rand, tot_rand),
                                            ("greedy π*", traj_opt, tot_opt)], 1):
        print(f"\nEpisode {i} with {label}: total reward = {tot:.3f}")
        for t, (s, a, r, sp) in enumerate(traj):
            print(f"  t={t:02d}  s={s}  a={a}  r={r:+.3f}  s'={sp}")

if __name__ == "__main__":
    run_demo()
