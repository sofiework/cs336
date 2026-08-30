import json, pandas as pd
from pathlib import Path


# results/*.json -> results.jsonl
# df -> latex

df = pd.read_json("results/results.jsonl", ) # DataFrame

cols = {
    "label": "label",
    "d_model": r"\texttt{d\_model}",
    "d_ff": r"\texttt{d\_ff}",
    "num_layers": r"\texttt{num\_layers}",
    "num_heads": r"\texttt{num\_heads}",
}

table = (
    df[list(cols)]
    .rename(columns=cols)
    .sort_values("label", key=lambda s: s.map({"small": 0, "medium": 1, "large": 2, "xl": 3, "10B": 4}))
)

latex = table.to_latex(index=False, escape=False, column_format="lrrrr", float_format="%.2f")

# write to latex file
Path("tables/model_sizes.tex").write_text(latex)