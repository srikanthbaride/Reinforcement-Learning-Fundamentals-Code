# Reinforcement Learning Fundamentals — Companion Code

This repository contains runnable Python code accompanying the book **"Reinforcement Learning Fundamentals: From Theory to Practice"**.

> Tip: Every chapter has its own folder with a short `README.md` and Python examples that mirror the book’s notation and figures.

## Contents

- `ch1_introduction/` — Toy examples and utilities used in the introduction.
- `ch2_rl_formulation/` — MDPs, policies, value functions, Bellman equations, policy evaluation, value iteration, grid world.
- `ch3_multi_armed_bandits/` — ε-greedy, UCB, Thompson Sampling (placeholders here; fill as chapter matures).
- `utils/` — Shared helpers (random seeds, plotting, gridworld helpers).

## Quickstart

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a Chapter 2 demo (GridWorld Value Iteration)
python ch2_rl_formulation/value_iteration.py
```

## Linking from LaTeX

In your LaTeX preamble:
```latex
\usepackage{hyperref}
\newcommand{\codelink}[2]{\href{#1}{\texttt{#2}}}
```

In Chapter 2:
```latex
\noindent\textbf{Companion Code:}
\codelink{https://github.com/YourUserName/Reinforcement-Learning-Fundamentals-Code/tree/main/ch2_rl_formulation}{GitHub (Chapter~2 Code)}.
```

## License

This project is licensed under the MIT License (see `LICENSE`).

