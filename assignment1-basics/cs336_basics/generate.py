import torch, json, einops

from cs336_basics.model import Transformer_LM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import load_checkpoint

vocab_file, merges_file = "tinystories_vocab.pkl", "tinystories_merges.pkl"


class Sequence:
    def __init__(self, token_ids: torch.Tensor):
        self.token_ids: list[int] = list(token_ids)
        self.num_prompt_token = len(self.token_ids)
        self.num_gen_token = 0

    def __len__(self):
        return self.num_prompt_token + self.num_gen_token

    def append(self, token_id: int):
        self.token_ids.append(token_id)

@torch.no_grad
def generate(prompt: str=None, max_tokens: int=128, temperature: float=1.0, p: float=0.9):
    # load model checkpoint, config
    with open("checkpoints/model_config.jsonl", "r") as f:
        config = json.load(f)
        # print(config.keys())

    src = "checkpoints/ckpt_4000.pt"
    device = config['device']

    model = Transformer_LM(
        config['d_model'],
        config['num_heads'],
        config['d_ff'],
        config['vocab_size'],
        config['num_layers'],
        theta=config['rope_theta'],
        max_seq_len=config['context_length'],
        device=device
    )
    load_checkpoint(src, model)

    print(f"cuda availability is {torch.cuda.is_available()}, model on {device}")

    # tokenize input -> wrap in Sequence
    tokenizer = Tokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])
    
    token_ids: list[int] = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, device=device)

    seq = Sequence(token_ids)

    # decoding single sequence
    predict = None
    while predict != "<|endoftext|>" and len(seq) < max_tokens:
        token_ids = torch.tensor(seq.token_ids, device=device)

        logits = model(token_ids) # [seq_len, v]
        logits = logits[-1, :].squeeze(0) # [1, v] -> [v]

        # softmax
        if temperature == 0:
            ids = torch.argmax(logits, dim=-1).item() # [1]
        else:
            # [vocab_size]
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # top-p
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True) # [v]
            accum = 0     # scalar
            keep = torch.zeros_like(probs, dtype=torch.bool, device=device) # [v]
            
            # exclusive i
            for i in range(probs.shape[-1]):
                # mask [: i] = 1 [i:] = 0
                keep[i] = accum < p # [v]
                accum = accum + sorted_probs[i]

            top_probs = torch.where(keep, sorted_probs, 0) # [b, 1, v]
            
            # renorm -> sum(top_prob) = 1
            top_probs = top_probs / torch.sum(top_probs, dim=-1, keepdim=True)

            i = torch.multinomial(top_probs, num_samples=1) # [1]
            ids: int = sorted_idx[i.item()].item()

        # update seq
        seq.append(ids)
        seq.num_gen_token += 1
            
        # logits decode -> token
        predict: str = tokenizer.decode([ids])
        print(predict, end="", flush=True) 

    return 

prompt = "Once upon a time, there is a princess called Lily,"

if __name__ == "__main__":
    print(prompt, end="", flush=True)
    generate(prompt)
    
    