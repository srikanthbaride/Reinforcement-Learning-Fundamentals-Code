# Chapter 3 — Multi-Armed Bandits

Implements ε-Greedy, UCB1, and Thompson Sampling on Bernoulli bandits.

## Run experiments
```bash
python -m ch3_multi_armed_bandits.experiments --K 10 --T 5000 --trials 50 --eps 0.1 --c 1.0
```

## Worked Examples
See `examples/` for scripts reproducing the numerical examples:
- Example 3.1: `ex1_regret_basic.py`
- Example 3.2: `ex2_epsilon_update.py`
- Example 3.3: `ex3_ucb_score.py`
- Example 3.4: `ex4_thompson_update.py`

## Tests
```bash
pytest -q ch3_multi_armed_bandits/tests
```
