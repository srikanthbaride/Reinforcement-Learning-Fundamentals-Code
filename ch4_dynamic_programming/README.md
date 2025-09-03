# Chapter 4 â€” Dynamic Programming (DP)

This folder contains minimal, well-tested NumPy implementations of the core DP algorithms:
- **Iterative Policy Evaluation**
- **Policy Iteration** (Howard, 1960)
- **Value Iteration** (Bellman optimality)

**Environment:** a deterministic 4Ã—4 GridWorld with an absorbing terminal goal at the top-right.

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
â”œâ”€â”€ __init__.py
â”œâ”€â”€ gridworld.py
â”œâ”€â”€ policy_evaluation.py
â”œâ”€â”€ policy_iteration.py
â”œâ”€â”€ value_iteration.py
â”œâ”€â”€ utils.py
â”œâ”€â”€ examples/
â”‚   â”œâ”€â”€ run_policy_iteration.py
â”‚   â””â”€â”€ run_value_iteration.py
â””â”€â”€ tests/
    â”œâ”€â”€ test_policy_evaluation.py
    â””â”€â”€ test_policy_and_value_iteration.py
```

