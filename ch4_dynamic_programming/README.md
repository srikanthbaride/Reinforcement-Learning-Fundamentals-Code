# Chapter 4 — Dynamic Programming

Implements **policy evaluation, policy improvement, policy iteration, and value iteration** for Markov Decision Processes with known dynamics.  
Includes convergence checks, numeric examples, and GridWorld demos.  

---

## ✅ Requirements

- Python ≥ 3.10
- `pip install -r requirements.txt` (use the repo-root `requirements.txt`)

---

## 🚀 Quickstart

```bash
# Run policy iteration demo
python -m ch4_dynamic_programming.examples.policy_iteration_demo

# Run value iteration demo
python -m ch4_dynamic_programming.examples.value_iteration_demo
```

---

## 📂 Layout

```
ch4_dynamic_programming/
├─ __init__.py
├─ dp.py                   # core DP algorithms: policy evaluation, improvement, iteration, value iteration
├─ gridworld.py            # GridWorld environment adapted for DP
├─ examples/
│  ├─ policy_iteration_demo.py
│  └─ value_iteration_demo.py
└─ tests/
   ├─ test_policy_evaluation.py
   ├─ test_policy_iteration.py
   └─ test_value_iteration.py
```

---

## 🧠 What’s Inside (Brief API)

### `dp.py`
- `policy_evaluation(P, R, policy, gamma=0.99, tol=1e-8, max_iters=10_000)`  
- `policy_improvement(Q)`  
- `policy_iteration(P, R, gamma=0.99, tol=1e-8, max_iters=10_000)`  
- `value_iteration(P, R, gamma=0.99, tol=1e-8, max_iters=10_000)`  

### `gridworld.py`
- `DPGridWorld` — tabular environment with full transition & reward matrices.  

---

## 🧪 Tests

```bash
pytest -q ch4_dynamic_programming/tests
```

Covers:
- Convergence of policy evaluation  
- Correctness of policy iteration  
- Optimality of value iteration  

---

## 📊 Notes

- GridWorld is small enough for exact DP solutions.  
- Demonstrates how Bellman equations can be solved recursively when the full model is available.  

---

## 🔗 Related

- Chapter 2 (RL Problem Formulation): MDPs, Bellman equations  
- Chapter 3 (Multi-Armed Bandits): exploration strategies without state transitions  
