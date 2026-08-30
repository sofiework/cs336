import torch
import timeit, argparse

from cs336_basics.model import Transformer_LM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy

from cssrc.config import ModelConfig
from cssrc.logger import Logger


"""
this script does benchmark of:
w warmup + n steps

timing:
for
for + back
for + back + opt (full)

"""



# hyper parameters: d_model, d_ff, num_layers, num_heads
SIZES = {
    "small": ModelConfig("small", 768, 3072, 12, 12),
    "medium": ModelConfig("medium", 1024, 4096, 24, 16),
    "large": ModelConfig("large", 1280, 5120, 36, 20),
    "xl": ModelConfig("xl", 2560, 10240, 32, 32),
    "10B": ModelConfig("10B", 4608, 12288, 50, 36),
}

WARMUP = 5 # for CUDA context, don't scale with model size
STEPS = 10
VOCAB = 10000
CTX_LEN = 512
BATCH = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    p = argparse.ArgumentParser()

    # fwd / fwd_bwd / fwd_bwd_opt
    p.add_argument("--mode", type=str, default="fwd")
    p.add_argument("--label", type=str, default="small")

    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--context_length", type=int, default=CTX_LEN)
    p.add_argument("--vocab_size", type=int, default=VOCAB)
    p.add_argument("--batch_size", type=int, default=BATCH)

    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--seed", type=int, default=0)

    # for optimizer
    p.add_argument("--max_lr", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    return p.parse_args()



def bench_mode_label(args):
    
    torch.manual_seed(args.seed)
    
    ### initiate model & opt
    config = SIZES[args.label]
    
    model = Transformer_LM(
        d_model=config.d_model,
        num_head=config.num_heads,
        d_ff=config.d_ff,
        vocab_size=args.vocab_size,
        num_layers=config.num_layers,
        max_seq_len=args.context_length,
        device=args.device
    )

    optimizer = AdamW(model.parameters(), args.max_lr, (args.beta1, args.beta2), args.adam_eps, args.weight_decay)

    log_file = f"results/bench_{args.label}.jsonl"
    logger = Logger(log_file, args, config)

    # randomize token IDs: [batch_size, context_length]
    token_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)


    def step():
        # time forward
        logits = model(token_ids)

        # time forward + back
        if args.mode == "fwd_bwd" or args.mode == "fwd_bwd_opt":
            # wait for all scheduled GPU kernel to finish
            loss = cross_entropy(logits, token_ids)
            optimizer.zero_grad()
            loss.backward()

            # time forward + back + opt
            if args.mode == "fwd_bwd_opt":
                optimizer.step()


    ### warmup CUDA context
    for _ in range(args.warmup):
        step()
    # sync after warmup
    if args.device == "cuda":
        torch.cuda.synchronize()

    ### n steps
    times = []
    
    for i in range(args.steps): # 10 measurament
        start = timeit.default_timer()
        step()

        if args.device == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
        print(f"running iter {i} of label {args.label}, mode {args.mode}")

    time_mean = sum(times) / len(times) # float scalar
    time_std = torch.std(torch.tensor(times)).item() # float scalar


    ### logging
    logger.log(time_mean, time_std)

def main():
    modes = ["fwd", "fwd_bwd", "fwd_bwd_opt"]
    labels = ["small", "medium", "large", "xl", "10B"]

    # parse once, and call bench on different label & mode
    args = get_args()
    for mode in modes:
        for label in labels:
            args.label = label
            args.mode = mode
            bench_mode_label(args)

            print(f"benchmarking done for label: {label}, mode: {mode}")

            # empty between test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()