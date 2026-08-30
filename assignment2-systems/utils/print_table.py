import json, pandas as pd
from pathlib import Path


# results/*.json -> results.jsonl
# df -> latex


# Gather
datas = []
"""
records = [
    {"mode": "fwd",         "label": "small",  "time_mean": 0.0121, "time_std": 0.0001},
    ...
]

"""

with (
    open("results/bench_small.jsonl") as small,
    open("results/bench_medium.jsonl") as medium,
    open("results/bench_large.jsonl") as large,
    open("results/bench_xl.jsonl") as xl,
):
    records = []
    pending = {}

    for lines in (small, medium, large, xl):
        for l in lines:
            d = json.loads(l)
            if "test_setup" in d:
                pending = {"mode": d["test_setup"]["mode"], "label": d["test_setup"]["label"]}
            elif "time_mean" in d:
                records.append({**pending, "time_mean": d["time_mean"], "time_std": d["time_std"]})
                pending = {}


# Make dataframe
df = pd.DataFrame(records)
table = df.pivot(index="mode", columns="label", values="time_std")
table = table[["small", "medium", "large", "xl"]]          # force column order
table = table.reindex(["fwd", "fwd_bwd", "fwd_bwd_opt"])   # force row order

# Write to latex file
# latex = table.to_latex(index=False, escape=False, column_format="lrrrr", float_format="%.2f")
# Path("results/model_sizes.tex").write_text(latex)

# Write to md
Path("results/model_std.md").write_text((table).to_markdown(floatfmt=".6f"))