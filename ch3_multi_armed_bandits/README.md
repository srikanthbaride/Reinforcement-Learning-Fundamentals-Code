# Chapter 3 — Multi-Armed Bandits

Implements **ε-Greedy**, **UCB1**, and **Thompson Sampling** strategies on Bernoulli bandits.  
Includes worked examples, experiments, and PyTest-based validation.

---

## 🚀 Run Experiments

```bash
python -m ch3_multi_armed_bandits.experiments --K 10 --T 5000 --trials 50 --eps 0.1 --c 1.0
```

Arguments:
- `K` — number of arms  
- `T` — time horizon (steps)  
- `trials` — number of independent runs  
- `eps` — exploration rate (for ε-greedy)  
- `c` — confidence level (for UCB1)  

---

## 📘 Worked Examples

Scripts reproducing the numerical examples from the chapter:

- **Example 3.1:** `ex1_regret_basic.py` — cumulative regret calculation  
- **Example 3.2:** `ex2_epsilon_update.py` — incremental update rule in ε-greedy  
- **Example 3.3:** `ex3_ucb_score.py` — UCB1 confidence bound score computation  
- **Example 3.4:** `ex4_thompson_update.py` — Bayesian update for Thompson Sampling  

Run them directly, e.g.:

```bash
python -m ch3_multi_armed_bandits.examples.ex1_regret_basic
```

---

## 🧪 Tests

```bash
pytest -q ch3_multi_armed_bandits/tests
```

Covers:
- Regret monotonicity  
- ε-greedy incremental update  
- UCB1 bound computation  
- Thompson Sampling’s posterior update  

---

## 📂 Layout

```
ch3_multi_armed_bandits/
├─ __init__.py
├─ bandits.py              # Bernoulli bandit environment
├─ strategies.py           # ε-greedy, UCB1, Thompson Sampling
├─ experiments.py          # CLI for running large-scale experiments
├─ examples/
│  ├─ ex1_regret_basic.py
│  ├─ ex2_epsilon_update.py
│  ├─ ex3_ucb_score.py
│  └─ ex4_thompson_update.py
└─ tests/
   ├─ test_bandits.py
   ├─ test_strategies.py
   └─ test_regret.py
```

---

## 🔗 Related

- **Chapter 2 — The RL Problem Formulation**: foundational MDP setup and Bellman equations  
- **Chapter 4 — Dynamic Programming**: full MDP solution methods (policy/value iteration)  
