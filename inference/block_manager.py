



class KVCacheNaive(nn.Module):
    def __init__(self, max_batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, 
                 dtype=torch.fp16, device=None):
        super().__init__()

        # [B, H_kv, N, D]
        cache_shape = (max_batch_size, )

        # register tensor
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))

    def 



class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)