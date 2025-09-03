# Chapter 2 â€” The RL Problem Formulation

Implements: MDP formalism, Bellman expectation & optimality, gridworld, greedy/Îµ-greedy, value iteration.
Includes numeric examples (5.23 and 4.58), demos, visualizations, and tests.

## Quickstart
```bash
python -m ch2_rl_formulation.examples.numeric_checks
python -m ch2_rl_formulation.examples.gridworld_demo
python -m ch2_rl_formulation.examples.plot_value_and_policy
```

## Layout
- `gridworld.py` â€” 4Ã—4 deterministic GridWorld.
- `evaluation.py` â€” policy evaluation (deterministic & stochastic), `q_from_v`, `greedy_from_q`.
- `policies.py` â€” deterministic & Îµ-greedy helpers.
- `value_iteration.py` â€” value iteration, extract greedy policy.
- `visualize.py` â€” single-plot, matplotlib-based visuals (no explicit colors).
- `examples/` â€” boxed examples and demos.
- `tests/` â€” sanity checks tied to the chapter.

