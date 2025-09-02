# Chapter 4 — Dynamic Programming (DP)

This folder contains minimal, well-tested NumPy implementations of the core DP algorithms:
- **Iterative Policy Evaluation**
- **Policy Iteration** (Howard, 1960)
- **Value Iteration** (Bellman optimality)

**Environment:** a deterministic 4×4 GridWorld with an absorbing terminal goal at the top-right.

## Run examples

```bash
python -m ch4_dynamic_programming.examples.run_policy_iteration
python -m ch4_dynamic_programming.examples.run_value_iteration
```

## Run tests

```bash
python -m pytest -q ch4_dynamic_programming/tests
```

## File map

```
ch4_dynamic_programming/
├── __init__.py
├── gridworld.py
├── policy_evaluation.py
├── policy_iteration.py
├── value_iteration.py
├── utils.py
├── examples/
│   ├── run_policy_iteration.py
│   └── run_value_iteration.py
└── tests/
    ├── test_policy_evaluation.py
    └── test_policy_and_value_iteration.py
```
