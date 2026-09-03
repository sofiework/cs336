import torch
import argparse, timeit

from cs336_basics.model import scaled_dot_product_attn
import cs336_basics.model
# from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy
from cssrc.logger import Logger
# from nsys_annt_attn import annotated_scaled_dot_product_attn

from cssrc.config import ModelConfig





# hyper parameters: d_model, d_ff, num_layers, num_heads
DFF, L, H = 1024, 16, 1 # not-in-use
SIZE = {
    16: ModelConfig("d16", 16, DFF, L, H),
    32: ModelConfig("d32", 32, DFF, L, H),
    64: ModelConfig("d64", 64, DFF, L, H),
    128: ModelConfig("d128", 128, DFF, L, H)
}

WARMUP = 5 # for CUDA context, don't scale with model size
STEPS = 100
BATCH = 8
CTX_LEN = 256
D = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    p = argparse.ArgumentParser()

    # fwd / fwd_bwd / fwd_bwd_opt
    p.add_argument("--mode", type=str, default="fwd")
    p.add_argument("--label", type=str, default="small")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--memory_out_name", type=str, default="memory_prof")

    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=BATCH)
    p.add_argument("--context_length", type=int, default=CTX_LEN)
    p.add_argument("--d_model", type=int, default=D)

    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()



def bench_mode_label(args):
    
    torch.manual_seed(args.seed)

    ### logger
    log_file = f"results/pyattn_{args.context_length}_D{args.d_model}"
    logger = Logger(log_file, args, SIZE[args.d_model])

    mask = torch.ones(args.context_length, args.context_length, dtype=bool).tril(diagonal=0).to(args.device)

    def step() -> torch.Tensor:
        # qkv shaep [B, seq_len, D]
        # requires_grad=True
        queries = torch.randn(args.batch_size, args.context_length, args.d_model, device=args.device, requires_grad=True)
        keys = torch.randn(args.batch_size, args.context_length, args.d_model, device=args.device, requires_grad=True)
        values = torch.randn(args.batch_size, args.context_length, args.d_model, device=args.device, requires_grad=True)

        return scaled_dot_product_attn(queries, keys, values, mask)


    ### mix_precision bf16

    import torch.cuda.nvtx as nvtx
    from contextlib import nullcontext

    ctx =  torch.autocast(device_type=args.device, dtype=torch.bfloat16) if args.use_bf16 else nullcontext()

   
    ### warmup CUDA context
    for _ in range(args.warmup):
        step()
    # sync after warmup
    if args.device == "cuda":
        torch.cuda.synchronize()


    fwd_time = []
    bwd_time = []

    ### profile n steps
    for _ in range(args.steps):

        # forward
        start = timeit.default_timer()
        
        logits = step()

        if args.device == "cuda":
            torch.cuda.synchronize()

        end = timeit.default_timer()
        fwd_memory = torch.cuda.memory.memory_allocated()

        fwd_time.append(end - start)



        # backward
        start = timeit.default_timer()
        logits.backward(torch.ones_like(logits))

        if args.device == "cuda":
            torch.cuda.synchronize()

        end = timeit.default_timer()
        bwd_time.append(end - start)
        

    mean_fwd_time = sum(fwd_time) / len(fwd_time)
    mean_bwd_time = sum(bwd_time) / len(bwd_time)
    
    logger.log_time_mem(mean_fwd_time, mean_bwd_time, fwd_memory)
    

def main():
    d_models = [16, 32, 64, 128]
    contexts = [256, 1024, 2048, 4096, 8192, 16384]

    # one config per nsys profile
    args = get_args()
    for d_model in d_models:
        for context_length in contexts:
            args.d_model = d_model
            args.context_length = context_length

            try:
                bench_mode_label(args)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM error: d_model = {d_model}, context_length = {context_length}")
            finally:

                # empty between test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    main()

