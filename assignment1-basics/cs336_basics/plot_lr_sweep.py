import matplotlib.pyplot as plt
import json

"""
fig1 - val loss / steps
ax_y1 = d["val_loss"]
ax_y2 = d["train_loss"]

ax_x = d["iter"]

fig2 - best val loss / max_lr
ax_y = min(d["val_loss"])

ax_x = max_lr
"""

# combine two tests

with open("experiments/lr_sweep_0.001.jsonl", "r") as f1:
    for _ in range(3):
        l1 = next(f1)
    d1 = json.loads(l1)
    print(d1.keys())

def plot():
    # fig1
    with open("experiments/lr_sweep_0.001.jsonl", "r") as f1, open("experiments/lr_sweep_0.003.jsonl", "r") as f3, open("experiments/lr_sweep_0.006.jsonl", "r") as f6, open("experiments/lr_sweep_0.009.jsonl", "r") as f9, open("experiments/lr_sweep_0.0001.jsonl", "r") as f01, open("experiments/lr_sweep_0.0003.jsonl", "r") as f03:
        x_steps = list(range(1000)) # [0, ... 999]
        x_val_steps = list(range(0, 1000, 200)) # [0, 200,...]
        y_train = [[] for _ in range(6)]
        y_val = [[] for _ in range(6)]

        # fig2
        y_min_val = [] # 4
        max_lr = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 9e-3]

        # max_lr = []

        for lines in zip(f01, f03, f1, f3, f6, f9):
            for n, l in enumerate(lines):
                d = json.loads(l)
                
                if "label" not in d:
                    y_train[n].append(d['train_loss'])
                    if d['val_loss'] != -1:
                        y_val[n].append(d['val_loss'])
                

        # get min val of each sweep
        for n in range(6):
            y_min_val.append(min(y_val[n]))

        # plot(x, y)
        # ax1 - train, ax2 - val
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        # ax4.set_visible(False)

        for n in range(4):
            ax1.plot(x_steps, y_train[n], label=f"max_lr={max_lr[n]}", linewidth=1)
            ax2.plot(x_val_steps, y_val[n], label=f"max_lr={max_lr[n]}", linewidth=1)
            ax3.plot(max_lr, y_min_val, label="")

            # best max_lr(1e-3) dropping with steps
            ax4.plot(x_steps, y_train[2], label=f"best max_lr={max_lr[2]}")

        ax1.set_title("lr_sweep" + "_train loss")
        ax2.set_title("lr_sweep" + "_val loss")
        ax2.set_title("min val loss / max_lr")

        ax1.set_xlabel("steps")
        ax1.set_ylabel("train loss")
        ax2.set_xlabel("steps")
        ax2.set_ylabel("val loss")
        ax3.set_xlabel("max_lr")
        ax3.set_ylabel("min val loss")
        
        ax1.legend()
        ax2.legend()
        ax3.legend()

        fig.savefig("experiments/maxmaxlr2.png")

plot()