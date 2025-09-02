from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from rldp.latex import grid_csv_to_tabular

def convert_one(csv_path: str, out_dir: str, caption: str | None, label: str | None,
                colfmt: str | None, float_format: str | None, index: bool,
                wrap_table: bool, round_digits: int | None,
                transpose: bool, suffix: str) -> str:
    tex = grid_csv_to_tabular(csv_path, caption, label, colfmt, float_format,
                              index, True, wrap_table, round_digits, transpose)
    name = Path(csv_path).stem + (suffix if suffix else "") + ".tex"
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(Path(out_dir) / name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    return out_path

def main():
    p = argparse.ArgumentParser(description="Convert CSV grids (values/policies) to LaTeX tables.")
    p.add_argument("inputs", nargs="+", help="CSV files or a directory (will convert all *.csv).")
    p.add_argument("--outdir", default="artifacts/latex", help="Output directory for .tex tables.")
    p.add_argument("--caption", default=None, help="Caption to use (optional).")
    p.add_argument("--label", default=None, help="LaTeX label (e.g., tab:vi-iterations).")
    p.add_argument("--colfmt", default=None, help="LaTeX column format, e.g., 'cccc'.")
    p.add_argument("--float-format", default=None, help="Python format, e.g., '.0f' or '{:.2f}'.")
    p.add_argument("--index", action="store_true", help="Include DataFrame index.")
    p.add_argument("--no-wrap", action="store_true", help="Emit only tabular (no table environment).")
    p.add_argument("--round", type=int, default=None, help="Round all numbers to N decimals.")
    p.add_argument("--transpose", action="store_true", help="Transpose before rendering.")
    p.add_argument("--suffix", default="", help="Append to output filename stem (e.g., '_nice').")
    args = p.parse_args()

    # Expand inputs: if a directory is given, take all CSVs in it
    files = []
    for item in args.inputs:
        pth = Path(item)
        if pth.is_dir():
            files.extend(str(p) for p in pth.glob("*.csv"))
        elif pth.suffix.lower() == ".csv":
            files.append(str(pth))
        else:
            print(f"Skipping non-CSV: {item}", file=sys.stderr)

    if not files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    created = []
    for csv in sorted(files):
        out = convert_one(
            csv_path=csv,
            out_dir=args.outdir,
            caption=args.caption,
            label=args.label,
            colfmt=args.colfmt,
            float_format=args.float_format,
            index=args.index,
            wrap_table=not args.no_wrap,
            round_digits=args.round,
            transpose=args.transpose,
            suffix=args.suffix,
        )
        created.append(out)
        print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
