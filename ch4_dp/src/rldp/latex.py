from __future__ import annotations
import os
import pandas as pd
from typing import Optional

def _latex_escape(text: str) -> str:
    if text is None:
        return ""
    return (text.replace("\\", r"\textbackslash{}")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("$", r"\$")
                .replace("#", r"\#")
                .replace("_", r"\_")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("~", r"\textasciitilde{}")
                .replace("^", r"\textasciicircum{}"))

def dataframe_to_tabular(df: pd.DataFrame,
                         caption: Optional[str] = None,
                         label: Optional[str] = None,
                         colfmt: Optional[str] = None,
                         float_format: Optional[str] = None,
                         index: bool = False,
                         booktabs: bool = True,
                         wrap_table: bool = True) -> str:
    if colfmt is None:
        cols = df.shape[1] + (1 if index else 0)
        colfmt = "c" * cols

    kwargs = {
        "index": index,
        "escape": False,
        "column_format": colfmt,
        "na_rep": "",
        "bold_rows": False
    }
    if float_format:
        fmt = float_format
        if "{" in fmt:
            kwargs["float_format"] = lambda x: fmt.format(x)
        else:
            kwargs["float_format"] = lambda x: f"{x:{fmt}}"

    latex_body = df.to_latex(**kwargs)
    if wrap_table:
        cap = f"\\caption{{{_latex_escape(caption)}}}\n" if caption else ""
        lab = f"\\label{{{label}}}\n" if label else ""
        return (
            "\\begin{table}[H]\n"
            "\\centering\n"
            + cap + lab + latex_body +
            "\\end{table}\n"
        )
    return latex_body

def grid_csv_to_tabular(csv_path: str,
                        caption: Optional[str] = None,
                        label: Optional[str] = None,
                        colfmt: Optional[str] = None,
                        float_format: Optional[str] = None,
                        index: bool = False,
                        booktabs: bool = True,
                        wrap_table: bool = True,
                        round_digits: Optional[int] = None,
                        transpose: bool = False) -> str:
    df = pd.read_csv(csv_path)
    if transpose:
        df = df.T
    if round_digits is not None:
        df = df.round(round_digits)
    return dataframe_to_tabular(df, caption, label, colfmt, float_format, index, booktabs, wrap_table)
