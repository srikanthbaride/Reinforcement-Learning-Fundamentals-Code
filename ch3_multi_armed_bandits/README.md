# Chapter 3 — Multi-Armed Bandits

This chapter introduces the **multi-armed bandit problem** — a simplified decision-making setting that captures the core trade-off between **exploration and exploitation**.  
We implement the main algorithms covered in the text.

## Files
- `epsilon_greedy.py` — ε-Greedy action selection.
- `ucb.py` — Upper Confidence Bound (UCB) algorithm.
- `thompson_sampling.py` — Thompson Sampling via Beta-Bernoulli updates.
- `plot_bandit_strategies.py` — visualize cumulative rewards/regret.

## Quick Start Demos

Run any bandit strategy directly from the repo root:

```bash
# ε-Greedy demo
python ch3_multi_armed_bandits/epsilon_greedy.py

# UCB demo
python ch3_multi_armed_bandits/ucb.py

# Thompson Sampling demo
python ch3_multi_armed_bandits/thompson_sampling.py
