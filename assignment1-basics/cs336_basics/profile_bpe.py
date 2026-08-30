import cProfile, pstats
from cs336_basics.tokenizer import train_bpe_parallel

cProfile.run('train_bpe_parallel("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])', "out.prof")
pstats.Stats("out.prof").sort_stats("tottime").print_stats(20)