import torch
import timeit, argparse

from cs336_basics.model import Transformer_LM
import cs336_basics.model
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy
from nsys_annt_attn import annotated_scaled_dot_product_attn

from cssrc.config import ModelConfig


"""
this script does nsys benchmark of:

3 model size:
"medium": ModelConfig("medium", 1024, 4096, 24, 16),
"large": ModelConfig("large", 1280, 5120, 36, 20),
"xl": ModelConfig("xl", 2560, 10240, 32, 32),

3 context length:
512, 1024, max_len = 2048 (a100, 80GB)

    activation memory: 
        2 * B * L * len^2 (qkt)
        34 * B * D * len
    parameter memory: (4 * 2 * D * D + 2 * 3 * D * DFF) * L (bf16)
        + grad: 2 * parameter (fp32)
        + opt: 4 * parameter (fp32)
    
    
mode:
fwd
fwd_bwd
fwd_bwd_opt

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
STEPS = 5
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
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--use_bf16", action="store_true")

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

    if args.nvtx:
        cs336_basics.model.scaled_dot_product_attn = annotated_scaled_dot_product_attn

    
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


    # randomize token IDs: [batch_size, context_length]
    token_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)


    import torch.cuda.nvtx as nvtx
    from contextlib import nullcontext

    ctx =  torch.autocast(device_type=args.device, dtype=torch.bfloat16) if args.use_bf16 else nullcontext()

    @nvtx.range("forward backward mode")
    def step():
        with nvtx.range("forward"):
            # time forward
            with ctx: # no-op if not use_bf16
                logits = model(token_ids)

        
        # time forward + back
        if args.mode == "fwd_bwd" or args.mode == "fwd_bwd_opt":
            with nvtx.range("forward + backward"):
                # wait for all scheduled GPU kernel to finish
                with ctx:
                    loss = cross_entropy(logits, token_ids)
                optimizer.zero_grad()
                loss.backward()

                
                # time forward + back + opt
                if args.mode == "fwd_bwd_opt":
                    with nvtx.range("forward + backward + optimizer"):
                        optimizer.step()


    ### warmup CUDA context
    for _ in range(args.warmup):
        step()
    # sync after warmup
    if args.device == "cuda":
        torch.cuda.synchronize()


    ### profile n steps
    torch.cuda.cudart().cudaProfilerStart() # nsys profile after warmup

    for i in range(args.steps):
        with nvtx.range(f"step_{i}"):
            step()

        if args.device == "cuda":
            torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()
    print(f"benchmarking done for label: {args.label}")
    

def main():
    modes = ["fwd", "fwd_bwd", "fwd_bwd_opt"]
    labels = ["small", "medium", "large", "xl"]
    contexts = [512, 1024, 2048]

    # one config per nsys profile
    args = get_args()
    bench_mode_label(args)

    # empty between test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


"""
bench 
fwd, fwd_bwd
use_bf16 on / off
all model size

fix context_length, batch_size

"""