from pathlib import Path
import regex as re
from collections import defaultdict, Counter
import os
from multiprocessing import Pool
import functools
import pickle
from collections.abc import Iterable
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries

"""
algorithms

1. string -> re.split() by special_tokens ->
    re.finditer() pre-tokenizer -> frequency table

    
2. merge most frequent pair ("", "") -> 1 vocab element

    vocab size = special_token + 256 + num_merged_pairs

    
3. take multiple merges -> vocab set


re.finditer(PAT, str) -> iterable
    # for m in re.finditer(PAT, str):
        m.group() # ' cat'
        m.span()  # (3, 7)
        m.start() # 3
        m.end()   # 7
    # list(re.finditer(PAT, str)) to make it list

re.split(PAT, str) -> list[str]
    # exclude spliter
re.split("(PAT)", str) -> list[str]
    # include spliter

re.escape("<|endoftext|>")
    # -> '<\\|endoftext\\|>'
    # re.split(re.escape("<|endoftext|>"), str)

"""

PAT =  r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# input corpus -> return (vocab, merges)
# vocab: {ID -> token bytes}
# merges: [(bytes, bytes)]
def train_bpe(
        input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    
    # dataset
    with open(input_path, "r", encoding="utf-8") as f: dataset = f.read()


    # special token
    splited: list[str] = re.split("|".join(re.escape(t) for t in special_tokens), dataset)

    # vocab (0-255, special token)
    vocab = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")

    # pre-tokenizer
    frequency: dict[tuple[bytes], int] = defaultdict(int)

    for sp in splited:
        for pretoken in re.finditer(PAT, sp):
            # pretoken.group() -> str
            frequency[tuple(bytes([b]) for b in pretoken.group().encode("utf-8"))] += 1

    # merges
    num_merges = 0
    merges = []

    pairs = defaultdict(int) # tuple[bytes, bytes] -> count
    pairs_pos = defaultdict(set) # tuple[bytes, bytes] -> set(tuple)
    for word in frequency:
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pairs[p] += frequency[word]
            pairs_pos[p].add(word)

    while num_merges < vocab_size - 256 - len(special_tokens):
        # find max pair
        max_pair = max(pairs.items(), key = lambda x: (x[1], x[0]))[0] # (count, lex)
        pos_set = pairs_pos[max_pair]

        # update merges
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        num_merges += 1

        # update pairs, frequency
        # by updating word tuple with merged bytes
        for word in list(pos_set):
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == max_pair:
                    new_word.append(max_pair[0] + max_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[-1])

            # remove pair in old word
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pairs[p] -= frequency[word]
                pairs_pos[p].discard(word)

            # find new_pair from new_word
            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i + 1])
                pairs[new_p] += frequency[word]
                pairs_pos[new_p].add(tuple(new_word))
            
            # update frequency
            frequency[tuple(new_word)] += frequency[word] # count
            del frequency[word]

        # remove max_pair
        del pairs[max_pair]
        del pairs_pos[max_pair]

    return (vocab, merges)


def pretoken(s: int, e: int, input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        f.seek(s)
        chunk = f.read(e - s).decode("utf-8", errors="ignore")

    # special token
    splited: list[str] = re.split("|".join(re.escape(t) for t in special_tokens), chunk)

    freq: dict[tuple[bytes], int] = defaultdict(int)
    for sp in splited:
        for pretoken in re.finditer(PAT, sp):
            # pretoken.group() -> str
            freq[tuple(bytes([b]) for b in pretoken.group().encode("utf-8"))] += 1

    return freq

class RevPair:
    __slots__ = ("p",)
    def __init__(self, p):
        self.p = p

    def __lt__(self, other):
        return self.p > other.p
    
def train_bpe_parallel(
        input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 

    num_process = os.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
        bounds = list(zip(boundaries[:-1], boundaries[1:]))

    # multiprocessing pre-tokenizer
    frequency: dict[tuple[bytes], int] = defaultdict(int)

    with Pool(processes=num_process) as pool:
        freq = pool.starmap(pretoken, [(s, e, input_path, special_tokens) for s, e in bounds])

        for d in freq:
            for k, v in d.items():
                frequency[k] += v
    
    # vocab (0-255, special token)
    vocab = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")


    # merges
    # optimization: max heap to avoid max(pairs.items()) traverse
    num_merges = 0
    merges = []

    pairs = defaultdict(int)     # tuple[bytes, bytes] -> count
    pairs_pos = defaultdict(set) # tuple[bytes, bytes] -> set(tuple[bytes])
    for word, cnt in frequency.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pairs[p] += cnt
            pairs_pos[p].add(word)

    # max heap
    heap = [(-c, RevPair(p), p) for p, c in pairs.items()]
    heapq.heapify(heap)

    def update_pairs_pos(p, count):
        pairs[p] -= count
        if pairs[p] <= 0:
            pairs.pop(p)
            pairs_pos.pop(p, None)
        else:
            heapq.heappush(heap, (-pairs[p], RevPair(p), p))

    while num_merges < vocab_size - 256 - len(special_tokens):
        # find max pair
        # max_pair = max(pairs.items(), key = lambda x: (x[1], x[0]))[0] # (count, lex)
        max_pair = None
        while heap:
            c, _, p = heapq.heappop(heap)
            if pairs.get(p, 0) == -c:
                max_pair = p
                break

        if max_pair is None:
            break
            
        pos_set = pairs_pos[max_pair]

        # update merges
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        num_merges += 1
        

        # update pairs, frequency
        # by updating word tuple with merged bytes
        for word in list(pos_set):
            new_word = []
            i = 0
            while i < len(word) - 1:
                # merge max_pair in word
                if (word[i], word[i + 1]) == max_pair:
                    new_word.append(max_pair[0] + max_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[-1])
            new_word = tuple(new_word)

            # remove pair in old word
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p in pairs:
                    pairs_pos[p].discard(word)
                    update_pairs_pos(p, frequency[word])

            # find new_pair from new_word
            for i in range(len(new_word) - 1):
                new_p = (new_word[i], new_word[i + 1])
                pairs_pos[new_p].add(new_word)
                update_pairs_pos(new_p, -frequency[word])
            
            # update frequency
            frequency[new_word] = frequency.get(new_word, 0) + frequency[word] # count
            del frequency[word]

        # remove max_pair
        pairs.pop(max_pair, None)
        pairs_pos.pop(max_pair, None)

    return (vocab, merges)


"""
encoding
1. pre-tokenize


"""
class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        self.rev_vocab = {b : i for i, b in vocab.items()}
        self.vocab = vocab

        self.merges = merges
        self.rank = {pair : i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens
        self.sorted_special = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, 
                   special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f: 
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pre-tokenize
        # special token
        if self.special_tokens:
            pat = "(" + "|".join(re.escape(t) for t in self.sorted_special) + ")"
            pretoken: list[str] = re.split(pat, text)
        else:
            pretoken = [text]

        sequence: list[int] = [] # token ID

        # pretoken: list[str], pre[str]
        for pre in pretoken:
            if self.special_tokens and pre in self.special_tokens:
                sequence.append(self.rev_vocab[pre.encode("utf-8")])
                continue

            # each loop: str -> list[bytes]
            for ch in re.finditer(PAT, pre):
                # ch.group(): str
                token: list[bytes] = [bytes([b]) for b in ch.group().encode("utf-8")]

                # merges
                while len(token) > 1:
                    best_i, best_rank = None, None
                    for i in range(len(token) - 1):
                        r = self.rank.get((token[i], token[i + 1]), None)

                        # has merges && (first | better merges)
                        if r is not None and (best_rank is None or r < best_rank):
                            best_i, best_rank = i, r

                    if best_i is None: # no merges
                        break

                    # update token        
                    i = best_i
                    token = token[:i] + [token[i] + token[i + 1]] + token[i + 2:]

                # vocab encode
                for b in token:
                    sequence.append(self.rev_vocab[b])

        return sequence

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        byt: bytes = b"".join(self.vocab[i] for i in ids)
        s: str = byt.decode("utf-8", errors="replace")
        return s
