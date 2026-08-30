import matplotlib.pyplot as plt
import json

"""
### fig1 - train & val loss / cumulative tokens, 
                              cumulative wall-clock seconds
x
y


### fig2 - best val loss / batch_size
x
y
    
"""



def plot():
    batch_size = [1, 4, 8, 16, 32, 48]
    x_val_steps = list(range(0, 1000, 200)) # [0, 200,...]

    x_time = []
    x_tokens = []
    y_train = []
    y_val = []

    for bs in batch_size:
        x_time_tmp = []
        x_tokens_tmp = []
        with open(f"experiments/batch_sweep_{bs}.jsonl", "r") as f:
            next(f)
            for l in f:
                d = json.loads(l)
                # ...

        x_time.append(x_time_tmp)
        x_tokens.append(x_tokens_tmp)
    
    with (
        open("experiments/batch_sweep_1.jsonl", "r") as f1,
        open("experiments/batch_sweep_4.jsonl", "r") as f4,
        open("experiments/batch_sweep_8.jsonl", "r") as f8,
        open("experiments/batch_sweep_16.jsonl", "r") as f16,
        open("experiments/batch_sweep_32.jsonl", "r") as f32,
        open("experiments/batch_sweep_48.jsonl", "r") as f48,
    ):
        # get x
        x_time = [[] for _ in range(6)]
        x_tokens = [[] for _ in range(6)]
        x_steps = [] # [[], []...] each batch have different steps

        l1, l4, l8, l16, l32, l48 = next(f1), next(f4), next(f8), next(f16), next(f32), next(f48)
        for l in (l1, l4, l8, l16, l32, l48):
            d = json.loads(l)
            n = d["meta"]["steps"]
            x_steps.append(list(range(n)))

        
        
        # train loss
        y_train = [[] for _ in range(6)]
        # val loss
        y_val = [[] for _ in range(6)]



        # for lines in zip(f1, f4, f8, f16, f32, f48):
        #     for n, l in enumerate(lines):
        #         d = json.loads(l)

        #         # cumulative training time
        #         x_time[n].append(x_time[n][-1] + d['wall_clock_time'])

        #         # cumulative token
        #         x_tokens[n].append(d['token_seen'])

        #         # train, val loss
        #         y_train[n].append(d['train_loss'])
        #         if d['val_loss'] != -1:
        #             y_val[n].append(d['val_loss'])
                

        # get min val of each sweep
        y_min_val = []

        for n in range(6):
            y_min_val.append(min(y_val[n]))

        # plot(x, y)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(18, 12))
        ax6.set_visible(False)

        for n in range(6):
            # train, val loss / tokens
            ax1.plot(x_tokens[n], y_train[n], label=f"batch_size={batch_size[n]}")
            ax2.plot(x_val_steps, y_val[n], label=f"batch_size={batch_size[n]}")

            # train, val loss / time
            ax3.plot(x_time[n], y_train[n], label=f"batch_size={batch_size[n]}")
            ax4.plot(x_val_steps, y_val[n], label=f"batch_size={batch_size[n]}")

            # min val loss / batch_size
            ax5.plot(batch_size, y_min_val, label=f"best batch_size=")

        ax1.set_title("train, val loss / tokens")
        ax2.set_title("train, val loss / time")
        ax3.set_title("batch_sweep" + "_val loss")
        ax4.set_title("min val loss / batch_size")

        ax1.set_xlabel("tokens")
        ax1.set_ylabel("train loss")
        ax2.set_xlabel("tokens")
        ax2.set_ylabel("val loss")

        ax3.set_xlabel("time")
        ax3.set_ylabel("train loss")
        ax4.set_xlabel("time")
        ax4.set_ylabel("val loss")

        ax5.set_xlabel("batch_size")
        ax5.set_ylabel("val loss")
        
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()

        fig.savefig("experiments/batch.png")

plot()