# log one .jsonl per run -> results/*.json

# git commit hash, GPU type, seed, wall clock

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

import json, argparse, math, torch
import pandas as pd
from pathlib import Path

from cssrc.config import ModelConfig

def device_info() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {
        "device": p.name,                          # 'NVIDIA H200'
        "device_mem_gb": round(p.total_memory / 1e9, 1),
        "device_count": torch.cuda.device_count(),
        "capability": f"{p.major}.{p.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

class Logger:
    def __init__(self, log_file, args: argparse.Namespace, model_config: ModelConfig):
        self.log_file: str = log_file # .json
        self.args = args
        self.config = model_config

        with open(self.log_file, "a") as out_file:
            # per test data
            per_test = {
                "test_setup": {
                    "mode": self.args.mode,
                    "label": self.args.label,
                    "warmup": self.args.warmup
                },
                "meta": {
                    "seed": self.args.seed,
                    "steps": self.args.steps,
                    "vocab_size": self.args.vocab_size,
                    "context_length": self.args.context_length,
                    "batch_size": self.args.batch_size
                },
                "device": device_info(),
                "model": {
                    "d_model": self.config.d_model,
                    "d_ff": self.config.d_ff,
                    "num_layers": self.config.num_layers,
                    "num_heads": self.config.num_heads
                }
                # "total_tokens": self.args.context_length * self.args.batch_size,
                # "AdamW": {
                #     "max_lr": self.args.max_lr,
                #     "beta1": self.args.beta1,
                #     "beta2": self.args.beta2,
                #     "eps": self.args.adam_eps,
                #     "weight_decay": self.args.weight_decay
                # }
            }
            json.dump(per_test, out_file)
            out_file.write("\n")

    def log(self, time_mean, time_std):
        with open(self.log_file, "a") as out_file:
            # per step data
            per_step = {
                # "iter": i,
                "time_mean": time_mean,
                "time_std": time_std
            }

            json.dump(per_step, out_file)
            out_file.write("\n")


    def log_time_mem(self, mean_fwd_time, mean_bwd_time, fwd_memory):
        with open(self.log_file, "a") as out_file:
            # per step data
            per_step = {
                "mean_fwd_time": mean_fwd_time,
                "mean_bwd_time": mean_bwd_time,
                "fwd_memory": fwd_memory
            }

            json.dump(per_step, out_file)
            out_file.write("\n")