from dataclasses import dataclass



@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int          # query heads
    n_kv_heads: int       # GQA: n_kv_heads <= n_heads, n_heads % n_kv_heads == 0

    d_ff: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    