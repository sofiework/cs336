# CS336 Assignment 2 (Systems) — Handout Catalog

Section map of `cs336_assignment2_systems.pdf` (48 pages), with every
`Problem (...)` block and its point value. Page numbers are PDF pages.

Total: **137 points** across 27 problems.

---

## 1 Assignment Overview — p1

| § | Title | p |
|---|---|---|
| 1.0.0.1 | What you will implement | 1 |
| 1.0.0.2 | What the code looks like | 1 |
| 1.0.0.3 | How to submit | 1 |

No problems.

---

## 2 Profiling and Benchmarking — p2 · **16 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 2.1 | Profiling | 2 | | |
| 2.1.1 | Setup — Importing your basics Transformer Model | 2 | | |
| 2.1.2 | Model Sizing | 2 | | |
| 2.1.3 | End-to-End Benchmarking | 3 | `benchmarking_script` | 4 |
| 2.1.4 | Nsight Systems Profiler | 4 | `nsys_profile` | 5 |
| 2.1.5 | Mixed Precision | 7 | `mixed_precision_accumulation` | 1 |
| | | 8 | `benchmarking_mixed_precision` | 2 |
| 2.1.6 | Profiling Memory | 9 | `memory_profiling` | 4 |

---

## 3 Single-GPU Memory — p10 · **4 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 3.1 | Autograd Residuals | 10 | | |
| 3.1.1 | Operator Fusion | 12 | | |
| 3.2 | Activation Checkpointing | 12 | | |
| 3.2.1 | Recomputation | 13 | `gradient_checkpointing` | 4 |

---

## 4 GPU Kernels — p15 · **29 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 4.1 | Optimizing Attention with FlashAttention-2 | 15 | | |
| 4.1.1 | Benchmarking PyTorch Attention | 15 | `pytorch_attention` | 2 |
| 4.2 | Benchmarking JIT-Compiled Attention | 16 | `torch_compile` | 2 |
| 4.2.1 | Example — Weighted Sum | 17 | | |
| 4.2.1.1 | Forward pass | 17 | | |
| 4.2.1.2 | Backward pass | 20 | | |
| 4.2.2 | FlashAttention-2 Forward Pass | 22 | `flash_forward` | 15 |
| 4.2.2.1 | Understanding inefficiencies in vanilla attention | 23 | | |
| 4.2.2.2 | Tiling | 23 | | |
| 4.2.2.3 | Recomputation | 23 | | |
| 4.2.2.4 | Operator fusion | 24 | | |
| 4.2.2.5 | Backward pass with recomputation | 24 | | |
| 4.2.2.6 | Details of the FlashAttention forward pass | 24 | | |
| 4.2.2.7 | Implementing the backward pass with recomputation | 28 | `flash_backward` | 5 |
| | | 28 | `flash_benchmarking` | 5 |
| 4.2.3 | OPTIONAL: Triton backward pass | 28 | | |

---

## 5 Distributed Data Parallel Training — p29 · **21 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 5.1 | Single-Node Distributed Communication in PyTorch | 30 | `distributed_communication_single_node` | 5 |
| 5.1.0.1 | Terminology | 31 | | |
| 5.1.1 | Best Practices for Benchmarking Distributed Applications | 32 | | |
| 5.2 | A Naïve Implementation of Distributed Data Parallel Training | 33 | `naive_ddp` | 5 |
| | | 33 | `naive_ddp_benchmarking` | 3 |
| 5.3 | Improving Upon the Minimal DDP Implementation | 34 | | |
| 5.3.1 | Reducing the Number of Communication Calls | 34 | `minimal_ddp_flat_benchmarking` | 2 |
| 5.3.2 | Overlapping Computation with Communication of Individual Parameter Gradients | 34 | `ddp_overlap_individual_parameters` | 5 |
| | | 36 | `ddp_overlap_individual_parameters_benchmarking` | 1 |

---

## 6 Optimizer State Sharding — p37 · **20 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 6 | Optimizer State Sharding | 37 | `optimizer_state_sharding` | 15 |
| | | 38 | `optimizer_state_sharding_accounting` | 5 |

---

## 7 Fully-Sharded Data Parallel — p38 · **20 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 7 | Fully-Sharded Data Parallel | 38 | `fsdp` | 15 |
| | | 39 | `fsdp_accounting` | 5 |

---

## 8 Analyzing Parallelism Strategies — p39 · **17 pts**

All pen-and-paper; no code.

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 8.1 | Communication Primitives | 40 | `alternate_ring_all_reduce` | 1 |
| 8.2 | Analyzing Data Parallel | 41 | `data_parallel_calcs` | 3 |
| 8.3 | Analyzing Fully Sharded Data Parallel | 42 | `fsdp_calcs` | 3 |
| 8.4 | Analyzing Tensor Parallel | 44 | `tp_calcs` | 4 |
| 8.5 | 2D Parallelism (FSDP + TP) | 45 | `fsdp_tp_calcs` | 6 |

---

## 9 Leaderboard — p46 · **10 pts**

| § | Title | p | Problem | Pts |
|---|---|---|---|---|
| 9 | Leaderboard | 46 | `leaderboard` (fastest training step) | 10 |

**Bibliography** — p48

---

