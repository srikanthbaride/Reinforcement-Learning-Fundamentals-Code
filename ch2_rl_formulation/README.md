# Chapter 2 — The RL Problem Formulation

Implements: MDP formalism, Bellman expectation & optimality, gridworld, greedy/ε-greedy, value iteration.
Includes numeric examples (5.23 and 4.58), demos, visualizations, and tests.

## Quickstart
```bash
python -m ch2_rl_formulation.examples.numeric_checks
python -m ch2_rl_formulation.examples.gridworld_demo
python -m ch2_rl_formulation.examples.plot_value_and_policy
```

## Layout
- `gridworld.py` — 4×4 deterministic GridWorld.
- `evaluation.py` — policy evaluation (deterministic & stochastic), `q_from_v`, `greedy_from_q`.
- `policies.py` — deterministic & ε-greedy helpers.
- `value_iteration.py` — value iteration, extract greedy policy.
- `visualize.py` — single-plot, matplotlib-based visuals (no explicit colors).
- `examples/` — boxed examples and demos.
- `tests/` — sanity checks tied to the chapter.
