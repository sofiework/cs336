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
                pending = {
                    "mode": d["test_setup"]["mode"], 
                    "label": d["test_setup"]["label"],
                    "warmup": d["test_setup"]["warmup"]
                }
            elif "time_mean" in d:
                records.append({**pending, "time_mean": d["time_mean"], "time_std": d["time_std"]})
                pending = {}


# Make dataframe 
df = pd.DataFrame(records)
# table = df.pivot(index="mode", columns="label", values="time_mean")
# table = table[["small", "medium", "large", "xl"]]          # force column order
# table = table.reindex(["fwd", "fwd_bwd", "fwd_bwd_opt"])   # force row order

sub = df[df["mode"] == "fwd_bwd_opt"]
table2 = (
    sub.pivot(index="warmup", columns="label", values="time_mean")
       [["small", "medium", "large", "xl"]]
       .sort_index()
)

# Write to md
# Path("results/model_mean.md").write_text((table).to_markdown(floatfmt=".1f"))

Path("results/model_warmup.md").write_text((table2 * 1000).to_markdown(floatfmt=".2f"))