import torch, argparse, typing, os, math, time, gc
import numpy as np
import json
import torch.nn as nn
import random

from cs336_basics.utils import data_loader, cross_entropy, gradient_clipping, lr_schedule, evaluate
from cs336_basics.model import Transformer_LM
from cs336_basics.optimizer import AdamW
from cs336_basics.logger import Logger


# PATH = "data/tinystories_train.npy"
# VAL_PATH = "data/tinystories_valid.npy"
PATH = "data/owt_train.npy"
VAL_PATH = "data/owt_val.npy"

STEPS = 1000

VOCAB = 10000
CTX_LEN = 256

EVAL_BATCH = 32
BATCH = 32
LAYERS = 4
NUM_HEADS = 16
D_MODEL = 512
D_FF = 1344



def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)


# return iteration: int
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


    
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--steps", type=int, default=STEPS) # total steps to train
    p.add_argument("--batch_size", type=int, default=BATCH)
    p.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup_bench", type=int, default=0)

    # model params
    p.add_argument("--vocab_size", type=int, default=VOCAB)
    p.add_argument("--context_length", type=int, default=CTX_LEN) # bs * ctx_len = 32 * 256 = 8K tokens
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--num_layers", type=int, default=LAYERS)
    p.add_argument("--num_heads", type=int, default=NUM_HEADS)
    p.add_argument("--d_ff", type=int, default=D_FF)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # optimizer
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    # schedule and gradient clipping
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_iters", type=int, default=None)
    p.add_argument("--cosine_cycle_iters", type=int, default=None)

    p.add_argument("--max_l2_norm", type=float, default=5.0)
    p.add_argument("--clip_eps", type=float, default=1e-8)

    # file path
    p.add_argument("--train_path", type=str, default=PATH) # train dataset file path
    p.add_argument("--dtype", type=str, default="uint16") # dataset dtype
    p.add_argument("--val_path", type=str, default=VAL_PATH) # val dataset file path

    # checkpoints
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints2")
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--resume_from", type=str, default=None) # the checkpoint to resume from

    # logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=200) # eval every _ batches
    p.add_argument("--eval_batches", type=int, default=20) # eval num_batches

    return p.parse_args()


import copy
def lr_sweep(args):
    os.makedirs("checkpoint_lr", exist_ok=True)

    for lr in [1e-4, 3e-4]:
        log_file = f"experiments/lr_sweep_{lr}.jsonl"
        # update in args
        run_args = copy.deepcopy(args)
        run_args.max_lr = lr
        run_args.checkpoint_dir = f"checkpoints_{lr}"

        os.makedirs(run_args.checkpoint_dir, exist_ok=True)

        main(run_args, log_file)


def batch_sweep(args):
    os.makedirs("checkpoints_batch", exist_ok=True)

    """
    batch_size limitation on 3060ti, 8GB

    param_size = (fp32)4B * (weights + grads + 2 AdamW states)
        num_param (mainly):
            embedding_matrix = v * dmodel
            lm_head = dmodel * v

            qkvo = 4 * dmodel * dmodel * layers
            FFN = 3 * dmodel * dff * layers

        total_num_weights: 
            10k * 512 * 2 + (16 * 512 * 512 + 12 * 512 * 1314) = 23M
        total_num_weights_size: 
            23M * 4B = 92MB
        total_num_param_size: 
            92MB * 4 = 368MB

    num_token = batch * context_length
        8GB - 1GB for CUDA = 7GB
        bytes_per_seq = 2 * 4B * (layers * heads * ctx^2 + ctx * vocab) = 54MB
            7GB / 54MB -> 120
    
    so batch_size sweep: [1, 4, 8, 16, 32, 64, 128]

    """
    for batch in [1, 4, 8, 16, 32, 48]:
        log_file = f"experiments/batch_sweep_{batch}.jsonl"
        # update in args
        run_args = copy.deepcopy(args)
        run_args.batch_size = batch
        run_args.checkpoint_dir = f"checkpoints_batch/checkpoints_{batch}"

        #### steps for differnet batch_size
        run_args.steps = math.ceil(48 * args.steps / batch)

        ## update warmup_iters, cosine_cycle_iters on current steps
        if run_args.warmup_iters is None:
            run_args.warmup_iters = math.ceil(run_args.steps / 20)
        if run_args.cosine_cycle_iters is None:
            run_args.cosine_cycle_iters = args.steps

        os.makedirs(run_args.checkpoint_dir, exist_ok=True)

        # run sweep with error check
        try:
            main(run_args, log_file)
        except torch.cuda.OutOfMemoryError as e:
            print(f"batch={batch} OOM: {e}")

        finally:
            # clear memory between sweeps
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()


def main(args, log_file):

    # args = get_args()
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Data loader

    # memory efficient load
    #   class memmap(np.ndarray)
    # train_data = np.memmap(args.train_path, mode="r", dtype=uint16)
    # val_data = np.memmap(args.val_path, mode="r")
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r")

    
    # Model & Optimizer
    model = Transformer_LM(
        d_model=args.d_model,
        num_head=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        theta=args.rope_theta,
        max_seq_len=args.context_length,
    ).to(args.device)
    # JIT-compiling
    compiled_model = torch.compile(model)

    optimizer = AdamW(model.parameters(), args.max_lr, (args.beta1, args.beta2), args.adam_eps, args.weight_decay)

    logger = Logger(log_file, args)

    # (Resume from)
    start_iter = load_checkpoint(args.resume_from, model, optimizer) + 1 if args.resume_from is not None else 0


    # Train loop
    for t in range(start_iter, args.steps):
        start = time.time()

        # data loader (each iter fetch small piece of data)
        inputs, targets = data_loader(train_data, args.batch_size, args.context_length, args.device)

        # lr schedule
        lr = lr_schedule(t, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # run model -> logits
        logits = compiled_model(inputs)
        loss = cross_entropy(logits, targets) # return Tensor

        # back propagate
        optimizer.zero_grad() # reset p.grad
        loss.backward() # compute and accumulate each p.grad

        # gradient descend
        grad_clip_norm = gradient_clipping(model.parameters(), args.max_l2_norm, args.clip_eps)
        optimizer.step() # update model params


        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.time()
        

        # eval
        if t % args.eval_interval == 0:
            val_loss = evaluate(compiled_model, val_data, args.eval_batch_size, 
                                args.context_length, args.device, args.eval_batches)
            print(f"iter {t}, train loss = {loss.item():.4f}, val loss = {val_loss:.4f}, lr = {lr:.2e}")
        else:
            val_loss = -1


        # log
        if t % args.log_interval == 0:
            print(f"iter {t}, train loss = {loss.item():.4f}, lr = {lr:.2e}")

        # save checkpoint
        if (t + 1) % args.checkpoint_interval == 0:
            file_path = os.path.join(args.checkpoint_dir, f"ckpt_{t}.pt")
            save_checkpoint(model, optimizer, t, file_path)

        ### logging
        
        logger.log(t, lr, val_loss, loss, end - start, grad_clip_norm)

if __name__ == "__main__":
    args = get_args()
    # lr_sweep(args)
    batch_sweep(args)
    pass
    
    # dump model config
    with open("checkpoints/model_config.jsonl", "w") as f:
        json.dump(vars(args), f)

    