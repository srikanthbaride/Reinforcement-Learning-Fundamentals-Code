# Chapter 6 — Temporal-Difference Learning (Code)

This folder accompanies Chapter 6 of your book and contains minimal, clean implementations of TD(0) and n-step TD prediction, a tiny Chain-3 environment, runnable demos, and pytest tests.

## Layout

```
ch6_td_learning/
├─ td0.py                # TD(0) prediction
├─ nstep_td.py           # n-step TD prediction (forward view)
├─ gridworld.py          # Chain3Env: A -> B -> C -> T with rewards {0, 0, +1}
├─ __init__.py
├─ examples/
│  ├─ td0_prediction_demo.py
│  └─ nstep_td_demo.py
└─ tests/
   └─ test_td_learning.py
```

A GitHub Actions workflow is included at `.github/workflows/ch6.yml` to run tests on push/PR.

## Quickstart

```bash
# from the repository root:
python -m ch6_td_learning.examples.td0_prediction_demo
python -m ch6_td_learning.examples.nstep_td_demo

# run tests (requires pytest)
pytest ch6_td_learning/tests -q
```

## API

### `td0_prediction(env, policy, gamma=0.99, alpha=0.1, episodes=500)`
Incremental TD(0) state-value prediction for a fixed policy.

### `nstep_td_prediction(env, policy, n=3, gamma=0.99, alpha=0.1, episodes=500)`
Forward-view n-step TD state-value prediction for a fixed policy.

### `Chain3Env()`
Deterministic chain: `A -> B -> C -> T`. Rewards `0, 0, +1`. Single dummy action (ignored).

## Expected values (Chain-3, γ=0.9)

True values under the deterministic policy:
- v(A) = γ³ = 0.729
- v(B) = γ² = 0.81
- v(C) = γ¹ = 0.9
- v(T) = 0
