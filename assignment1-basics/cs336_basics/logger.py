"""
general per test

"steps": int = STEPS
"model_param": int
    PATH = ""
    VOCAB = 10000
    CTX_LEN = 256

    LAYERS = 4
    BATCH = 32
    NUM_HEADS = 16
    D_MODEL = 512
    D_FF = 1344

    max_lr = 1e-5
    min_lr = 1e-3
    warmup_iters = math.ceil(STEPS / 20)
    cosine_cycle_iters = STEPS
"total_tokens": int = BATCH * STEPS * CTX_LEN

"AdamW": {
    betas: float
    eps: float
    weight_decay: float
}
"grad_clip": {
    "clip_eps": float
}
    


log per step

"iter": int
"val_loss": list[float]
"train_loss": list[float]
"train_val_ratio": float

"lr": float
"grad_clip_norm": float

"wall_clock_time": float
"token_seen": int = BATCH * (step + 1) * CTX_LEN

"""

import json, argparse, math


class Logger:

    def __init__(self, log_file, args: argparse.Namespace):
        self.log_file: str = log_file
        self.args = args

        with open(self.log_file, "w") as out_file:
            # per test data
            per_test = {
                "label": f"batch_sweep with batch = {self.args.batch_size}",
                "meta": {
                    "steps": self.args.steps,
                    "path": self.args.train_path,
                    "vocab_size": self.args.vocab_size,
                    "context_length": self.args.context_length,
                    "num_layers": self.args.num_layers,
                    "batch_size": self.args.batch_size,
                    "num_heads": self.args.num_heads,
                    "d_model": self.args.d_model,
                    "d_ff": self.args.d_ff,
                    "max_lr": self.args.max_lr,
                    "min_lr": self.args.min_lr,
                    "warmup_iters": self.args.warmup_iters,
                    "cosine_cycle_iters": self.args.cosine_cycle_iters
                },
                "total_tokens": self.args.batch_size * self.args.steps * self.args.context_length,
                "AdamW": {
                    "beta1": self.args.beta1,
                    "beta2": self.args.beta2,
                    "eps": self.args.adam_eps,
                    "weight_decay": self.args.weight_decay
                },
                "grad_clip": {
                    "clip_eps": self.args.clip_eps
                }
            }
            json.dump(per_test, out_file)
            out_file.write("\n")

    def log(self, iter_id, lr, val_loss, loss, time, grad_clip_norm):
        with open(self.log_file, "a") as out_file:
            # per iter data
            per_step = {
                "iter": iter_id,
                "val_loss": val_loss,
                "train_loss": loss.item(),
                # "train_val_diff": loss.item() - val_loss if val_loss != -1 else None,
                "lr": lr,
                "grad_clip_norm": grad_clip_norm,
                "wall_clock_time": time,
                "token_seen": self.args.batch_size * (iter_id + 1) * self.args.context_length
            }

            json.dump(per_step, out_file)
            out_file.write("\n")
                