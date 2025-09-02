# Chapter 4 – Dynamic Programming (DP) Code

This repo contains clean, runnable reference code to accompany **Chapter 4: Dynamic Programming Approaches**.

## Contents

- `src/rldp/dp.py` – Policy Evaluation, Policy Iteration, Value Iteration
- `src/rldp/gridworld.py` – Simple deterministic GridWorld (4×4, 6×6)
- `src/rldp/latex.py` – CSV → LaTeX table helper (booktabs-ready)
- `examples/generate_artifacts.py` – Reproduces tables/plots for the chapter
- `examples/csv_to_latex.py` – Convert CSV matrices to LaTeX tables
- `Makefile` – Convenience targets
- `requirements.txt` – Python deps

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Generate artifacts for 4×4 and 6×6 worlds
python examples/generate_artifacts.py --env 4x4 --outdir artifacts/ch4_4x4
python examples/generate_artifacts.py --env 6x6 --outdir artifacts/ch4_6x6

# Convert CSV → LaTeX tabular (booktabs)
python examples/csv_to_latex.py artifacts/ch4_4x4/vi_values_4x4_k2.csv   --outdir artifacts/latex   --caption "Value iteration estimates (k=2) on the $4\\times4$ gridworld."   --label tab:vi-4x4-k2   --float-format ".0f"
```

Then include in LaTeX:

```latex
\usepackage{booktabs} % in preamble
% ...
\input{artifacts/latex/vi_values_4x4_k2.tex}
```

## Make targets

```bash
make ch4-artifacts    # build default artifacts
make ch4-tables       # CSV → LaTeX for default directory
```

## License

MIT for the code snippets here. Attribution appreciated.
