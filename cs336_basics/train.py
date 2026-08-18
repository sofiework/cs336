import torch, argparse, typing, os
import numpy as np
import torch.nn as nn

from cs336_basics.utils import data_loader, cross_entropy, gradient_clipping, lr_schedule
from cs336_basics.model import Transformer_LM
from cs336_basics.optimizer import AdamW




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

@torch.no_grad()
def evaluate(model: Transformer_LM, val_data: np.ndarray, 
             batch_size: int, context_length: int, device=None, num_batches=20) -> float:
    model.eval()
    total = 0.0 # total loss on eval data

    for _ in range(num_batches):
        inputs, targets = data_loader(val_data, batch_size, context_length, device)
        logits = model(inputs)
        total += cross_entropy(logits, targets).item()

    model.train()
    return total / num_batches # avg over num_batches
    

    
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--steps", type=int, required=True) # total steps to train
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # model params
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, required=True)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--num_heads", type=int, required=True)
    p.add_argument("--d_ff", type=int, required=True)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # optimizer
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    # schedule and gradient clipping
    p.add_argument("--max_lr", type=float, required=True)
    p.add_argument("--min_lr", type=float, required=True)
    p.add_argument("--warmup_iters", type=int, required=True)
    p.add_argument("--cosine_cycle_iters", type=int, required=True)

    p.add_argument("--max_l2_norm", type=float, default=1.0)
    p.add_argument("--clip_eps", type=float, default=1e-8)

    # file path
    p.add_argument("--train_path", type=str, required=True) # train dataset file path
    p.add_argument("--dtype", type=str, default="uint16") # dataset dtype
    p.add_argument("--val_path", type=str, required=True) # val dataset file path

    # checkpoints
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--resume_from", type=str, default=None) # the checkpoint to resume from

    # logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=200) # eval every _ batches
    p.add_argument("--eval_batches", type=int, default=20) # eval num_batches

    return p.parse_args()


def main():

    args = get_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # Data loader

    # memory efficient load
    train_data = np.load(args.train_path, mmap_mode="r") # memory-mapped .npy: dtype read from header
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

    optimizer = AdamW(model.parameters(), args.max_lr, (args.beta1, args.beta2), args.adam_eps, args.weight_decay)

    # (Resume from)
    start_iter = load_checkpoint(args.resume_from, model, optimizer) + 1 if args.resume_from is not None else 0


    # Train loop
    for t in range(start_iter, args.steps):

        # data loader (each iter fetch small piece of data)
        inputs, targets = data_loader(train_data, args.batch_size, args.context_length, args.device)

        # lr schedule
        lr = lr_schedule(t, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # run model -> logits
        logits = model(inputs)
        loss = cross_entropy(logits, targets) # return Tensor

        # backpropagate
        optimizer.zero_grad() # reset p.grad
        loss.backward() # compute and accumulate each p.grad

        # gradient descend
        gradient_clipping(model.parameters(), args.max_l2_norm, args.clip_eps)
        optimizer.step() # update model params

        # log
        if t % args.log_interval == 0:
            print(f"iter {t}, train loss = {loss.item():.4f}, lr = {lr:.2e}")

        # eval
        if t % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, 
                                args.context_length, args.device, args.eval_batches)
            print(f"iter {t}, train loss = {loss.item():.4f}, val loss = {val_loss:.4f}, lr = {lr:.2e}")

        
        # save checkpoint
        if t % args.checkpoint_interval == 0:
            file_path = os.path.join(args.checkpoint_dir, f"ckpt_{t}.pt")
            save_checkpoint(model, optimizer, t, file_path)

if __name__ == "__main__":
    main()