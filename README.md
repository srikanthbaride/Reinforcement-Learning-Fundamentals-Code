# Reinforcement Learning Fundamentals — Companion Code

This repository contains runnable Python code accompanying the book **"Reinforcement Learning Fundamentals: From Theory to Practice"**.

> Tip: Every chapter has its own folder with a short `README.md` and Python examples that mirror the book’s notation and figures.

## Contents

- [`ch2_rl_formulation/`](ch2_rl_formulation/README.md) — MDPs, policies, value functions, Bellman equations, policy evaluation, value iteration, grid world.
- [`ch3_multi_armed_bandits/`](ch3_multi_armed_bandits/README.md) — ε-greedy, UCB, Thompson Sampling.
- [`utils/`](utils/) — Shared helpers (random seeds, plotting, gridworld helpers).

## Quickstart

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a Chapter 2 demo (GridWorld Value Iteration)
python ch2_rl_formulation/value_iteration.py

# 4) Run the Chapter 2 Random MDP demo
python ch2_rl_formulation/demo_random_mdp.py

# 5) Run Chapter 3 bandit demos
python ch3_multi_armed_bandits/epsilon_greedy.py
python ch3_multi_armed_bandits/ucb.py
python ch3_multi_armed_bandits/thompson_sampling.py
