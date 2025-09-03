# Chapter 2 — The RL Problem Formulation

Implements: **MDP** formalism, **Bellman expectation & optimality** equations, a 4×4 **GridWorld** environment, **greedy / ε-greedy** policies, and **value iteration**.  
Includes numeric examples (5.23 and 4.58), demos, visualizations, and tests aligned with the textbook.

---

## ✅ Requirements

- Python ≥ 3.10
- `pip install -r requirements.txt` (use the repo-root `requirements.txt`)

> Tip: Create and activate a virtual environment before installing.

---

## 🚀 Quickstart

```bash
# Numeric checks for examples 5.23 and 4.58
python -m ch2_rl_formulation.examples.numeric_checks

# GridWorld demo: evaluate a policy, compute Q, and act greedily
python -m ch2_rl_formulation.examples.gridworld_demo

# Plot values and a derived greedy policy (matplotlib, no explicit colors)
python -m ch2_rl_formulation.examples.plot_value_and_policy
```

---

## 📂 Layout

```
ch2_rl_formulation/
├─ __init__.py
├─ gridworld.py           # 4×4 deterministic GridWorld (tabular P, R)
├─ evaluation.py          # policy_evaluation(), q_from_v(), greedy_from_q()
├─ policies.py            # deterministic & ε-greedy policies
├─ value_iteration.py     # value_iteration(), extract greedy policy
├─ visualize.py           # minimal matplotlib plots (no fixed color maps)
├─ examples/
│  ├─ numeric_checks.py
│  ├─ gridworld_demo.py
│  └─ plot_value_and_policy.py
└─ tests/
   ├─ test_gridworld.py
   ├─ test_evaluation.py
   ├─ test_policies.py
   └─ test_value_iteration.py
```

---

## 🧠 What’s Inside (Brief API)

### `gridworld.py`
- `GridWorld4x4(step_reward=-1.0, goal=(0, 3))`
  - Attributes: `S` (states), `A` (actions), `P` (S×A×S′), `R` (S×A), helpers for indexing.

### `policies.py`
- `deterministic_policy(mapping_or_array)`  
- `epsilon_greedy_policy(Q, epsilon=0.1)`  

### `evaluation.py`
- `policy_evaluation(P, R, policy, gamma=0.99, tol=1e-8, max_iters=10_000)`  
- `q_from_v(P, R, V, gamma=0.99)`  
- `greedy_from_q(Q)`  

### `value_iteration.py`
- `value_iteration(P, R, gamma=0.99, tol=1e-8, max_iters=10_000)`  
- Returns `(V*, π*)` where `π*` is greedy w.r.t. `V*`.  

### `visualize.py`
- `plot_values(V, shape=(4,4))`  
- `plot_policy(pi, shape=(4,4))`  
- Uses matplotlib with default styles (no explicit colors set).  

---

## 🧪 Tests

```bash
pytest -q ch2_rl_formulation/tests
```

- Covers grid dynamics, policy evaluation convergence, greedy extraction, and value iteration optimality.

---

## 📊 Reproducibility Notes

- Matrices `P` and `R` are tabular numpy arrays (no randomness in dynamics).  
- Examples 5.23 and 4.58 match the book’s numerics (tolerances set in tests).  
- Plots avoid explicit color selection to keep CI/headless rendering consistent.  

---

## 🔗 Related

- Chapter 3 (Multi-Armed Bandits): action–selection strategies under uncertainty  
- Chapter 4 (Dynamic Programming): exact solutions with full model knowledge  
