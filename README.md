# Reinforcement Learning Fundamentals — Companion Code

[![Python (Chapters)](https://github.com/srikanthbaride/Reinforcement-Learning-Fundamentals-Code/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/srikanthbaride/Reinforcement-Learning-Fundamentals-Code/actions/workflows/python-tests.yml)


This repository hosts **chapter-wise companion code** for the book *Reinforcement Learning Fundamentals: From Theory to Practice*.  
It provides clean, minimal, and well-tested implementations of key reinforcement learning concepts.

---

## 📂 Repository Structure

```
rl-fundamentals-code/
├─ ch2_rl_formulation/              # Chapter 2: The RL Problem Formulation
│  ├─ gridworld.py                  # 4x4 GridWorld MDP (tabular P,R builder)
│  ├─ evaluation.py                 # Policy evaluation, q_from_v(), greedy_from_q()
│  ├─ policies.py                   # Deterministic & ε-greedy policies
│  ├─ value_iteration.py            # Bellman optimality, value iteration
│  ├─ examples/                     # Numeric examples, GridWorld demo, plotting
│  └─ tests/                        # Pytest-based checks for chapter numbers
│
├─ ch3_multi_armed_bandits/         # Chapter 3: Multi-Armed Bandits
│  ├─ bandits.py                    # Bernoulli & Gaussian bandit environments
│  ├─ epsilon_greedy.py             # Sample-average ε-greedy agent
│  ├─ ucb.py                        # UCB1 agent (with tunable exploration constant)
│  ├─ thompson.py                   # Beta–Bernoulli Thompson Sampling agent
│  ├─ experiments.py                # Run algorithms, generate regret plots
│  ├─ plots/                        # Saved figures (regret_bernoulli.png, etc.)
│  └─ tests/                        # Regression tests (ordering, sublinear regret)
│
├─ utils/                           # Shared helper utilities (future use)
│
├─ .github/workflows/               # CI: runs pytest on every push/PR
│  └─ python-tests.yml
│
├─ requirements.txt                 # Global dependencies (numpy, matplotlib, pytest)
├─ requirements_ch2.txt             # Chapter 2–specific dependencies
├─ requirements_ch3.txt             # Chapter 3–specific dependencies
└─ README.md                        # Project overview + usage

```

---

## 🚀 Getting Started

Clone this repository:

```bash
git clone https://github.com/srikanthbaride/Reinforcement-Learning-Fundamentals-Code.git
cd Reinforcement-Learning-Fundamentals-Code
```

Set up a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scriptsctivate    # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ✅ Running Tests

To run all tests:

```bash
python -m pytest -q
```

To run only Chapter 2 tests:

```bash
python -m pytest -q ch2_rl_formulation/tests
```

---

## 🧪 Examples

Run numeric checks for Chapter 2:

```bash
python -m ch2_rl_formulation.examples.numeric_checks
```

Run the GridWorld demo:

```bash
python -m ch2_rl_formulation.examples.gridworld_demo
```

---

## ⚙️ Continuous Integration

- GitHub Actions (`.github/workflows/python-tests.yml`) automatically run tests for all chapters on every push and pull request.
- This ensures correctness and reproducibility of the examples.

---

## 📖 License

MIT License © 2025 — Srikanth Baride
