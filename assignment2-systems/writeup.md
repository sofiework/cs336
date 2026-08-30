# CS336 Assignment 2 (Systems) — Written Answers

Scaffold of every problem in the handout that requires a **written** response
(prose, table, plot, screenshot, or math derivation). Pure-coding problems
(`naive_ddp`, `ddp_overlap_individual_parameters`, `optimizer_state_sharding`,
`fsdp`, `flash_forward`, `flash_backward`) are omitted — they're verified by
`pytest`, not by a writeup.

> Note: your stated focus is the §4.2 Triton sprint. §2.1.4–§8 below are
> included for completeness; fill in only what you actually run.

---

## §2.1.3 — `benchmarking_script` (4 pts)

**(b)** Time forward / backward / optimizer step for the Table 1 model sizes; 5 warm-up, 10 measurement steps (mean + std). How long is a forward pass? A backward pass? Is the std small or is there high variability?
*Deliverable: 1–2 sentence response with your timings.*

**Answer:**

<sub>RTX 3060 Ti (8 GB), fp32, context 512, batch 4, 5 warm-up / 10 measured steps.</sub>

| size | d_model | n_layers | forward (ms) | forward+backward (ms) | full step (ms) |
| --- | --- | --- | --- | --- | --- |
| small | 768 | 12 | 96.39 ± 0.67 | 309.99 ± 0.60 | 353.36 ± 0.51 |
| medium | 1024 | 24 | 293.37 ± 0.93 | — | — |
| large | 1280 | 36 | 629.82 ± 2.19 | — | — |
| xl | 1600 | 48 | — | — | — |
| 2.7B | 2560 | 32 | — | — | — |

<sub>— = run aborted before completing due to a host NVIDIA driver/library version mismatch (kernel 595.71.05 vs userspace NVML 595.84), **not** OOM. Reboot to reload the matching kernel module, then re-run to fill these cells.</sub>

**(c)** Repeat with **no** warm-up. How does it change the results and why? Then try 1–2 warm-up steps — why might the result still differ?
*Deliverable: 2–3 sentence response.*

**Answer:**

---

## §2.1.4 — `nsys_profile` (5 pts)

*(Profile fwd / bwd / optimizer with 2 model sizes from Table 1 and 3 power-of-two context lengths > 128.)*

**(a)** Total time of the forward pass — does it match your earlier Python-stdlib timing?
*Deliverable: 1–2 sentence response.*

**Answer:**

**(b)** Which CUDA kernel takes the most cumulative GPU time in the forward pass? How many times is it invoked per forward pass? Same kernel for forward+backward?
*Deliverable: 1–2 sentence response.*

**Answer:**

**(c)** Besides matmuls, what other kernels take non-trivial CUDA runtime in the forward pass?
*Deliverable: 1–2 sentence response.*

**Answer:**

**(d)** Profile one full training step (fwd + loss + bwd + AdamW step). How does the fraction of time in matmul change vs. inference-only? What about other kernels?
*Deliverable: 1–2 sentence response.*

**Answer:**

**(e)** Compare softmax vs. matmul runtime inside self-attention (forward). How does the runtime difference compare to the FLOP difference?
*Deliverable: 1–2 sentence response.*

**Answer:**

---

## §2.1.5 — Mixed Precision

### `mixed_precision_accumulation` (1 pt)

Run the given fp32 / fp16 accumulation snippets and comment on the accuracy of each result.
*Deliverable: 2–3 sentence response.*

**Answer:**

### `benchmarking_mixed_precision` (2 pts)

**(a)** For the `ToyModel` under FP16 autocast, give the dtype of: model parameters (in autocast), output of `fc1`, output of `ln` (LayerNorm), predicted logits, loss, and gradients.
*Deliverable: the data type for each component listed.*

**Answer:**
- model parameters:
- `fc1` output:
- `ln` output:
- logits:
- loss:
- gradients:

**(b)** Which parts of LayerNorm are sensitive to mixed precision? If using BF16 instead of FP16, do we still need to treat LayerNorm differently? Why / why not?
*Deliverable: 2–3 sentence response.*

**Answer:**

**(c)** Add a BF16 mixed-precision option; time fwd + bwd with and without it for each Table 1 size. Compare, and comment on trends as model size grows.
*Deliverable: 2–3 sentence response with timings and commentary.*

**Answer:**

---

## §2.1.6 — `memory_profiling` (4 pts)

*(xl model, context lengths 128 and 2048.)*

**(a)** Memory profile of xl for forward-only vs. a full training step — what do the timelines look like, and can you tell which stage is running from the peaks?
*Deliverable: two "Active memory timeline" images (forward, full step) + 2–3 sentence response.*

**Answer:**

**(b)** Peak memory per context length for a forward pass, and for a full training step.
*Deliverable: a table with two numbers per context length.*

**Answer:**

| context length | forward peak | full-step peak |
| --- | --- | --- |
| 128 |  |  |
| 2048 |  |  |

**(c)** Peak memory of xl under mixed precision (forward + full step) — does mixed precision significantly change memory?
*Deliverable: 2–3 sentence response.*

**Answer:**

**(d)** Size (in MiB) of one residual-stream activation tensor for xl at single precision — show the derivation.
*Deliverable: 1–2 sentence response with your derivation.*

**Answer:**

**(e)** In memory_viz, reduce the "Detail" level: what is the size of the largest allocations shown, and where (stack trace) do they come from?
*Deliverable: 1–2 sentence response.*

**Answer:**

**(f)** Using Nsight memory flags + PyTorch NVTX labels, determine how much memory a single `TransformerBlock` saves for backward (residuals). Note the 5 largest contributing operations and their % of the total.
*Deliverable: Nsight screenshots + 1–2 paragraph response.*

**Answer:**

---

## §3.1 — `gradient_checkpointing` (4 pts)

**(a)** Ignoring compute cost, what checkpointing strategy minimizes peak activation memory for N stacked blocks? Describe the arrangement (code sketch OK) and give the asymptotic peak memory and compute as a function of N.
*Deliverable: 3–5 sentence description + asymptotic peak memory + short code sketch.*

**Answer:**

**(b)** For xl, batch 4, seq len 2048, with budget for only **one** recomputation step (no nested checkpoints), what block size best reduces peak memory? Validate by profiling; compare the next-smaller and next-larger block sizes.
*Deliverable: 3–5 sentence reasoning + measured peak memory for your strategy.*

**Answer:**

---

## §4.1.1 — `pytorch_attention` (2 pts)

Benchmark attention (batch 8, no heads) over d_model ∈ {16,32,64,128} × seq len ∈ {256,1024,4096,8192,16384}: time 100 fwd, measure pre-backward memory, time 100 bwd (warm up + synchronize). At what size does it OOM? Do the memory accounting for a smallest OOM config. How does memory-saved-for-backward scale with seq len, and how would you eliminate it?
*Deliverable: a table of timings, your memory-usage calculations, and a 1–2 paragraph response.*

**Answer:**

---

## §4.2 — `torch_compile` (2 pts)

**(a)** Add a `torch.compile`d version of your attention; compare to the uncompiled version (same config as `pytorch_attention`).
*Deliverable: a table comparing compiled vs. uncompiled forward and backward timings.*

**Answer:**

**(b)** Compile the whole Transformer in your end-to-end script. How does forward-pass performance change? What about combined fwd+bwd+optimizer?
*Deliverable: a table comparing vanilla vs. compiled Transformer.*

**Answer:**

---

## §4.2.2 — `flash_benchmarking` (5 pts)

Compare your Triton FlashAttention-2 against the PyTorch attention implementation.
*Deliverable: a table reporting forward, backward, and end-to-end latencies for both.*

**Answer:**

---

## §5.1 — `distributed_communication_single_node` (5 pts)

Benchmark all-reduce (single node, multi-process) across data sizes {1MB, 10MB, 100MB, 1GB} × {2, 4, 6} GPUs.
*Deliverable: plot(s)/table(s) comparing the settings + 2–3 sentences on how the factors interact.*

**Answer:**

---

## §5.2 — `naive_ddp_benchmarking` (3 pts)

Benchmark naïve DDP (per-parameter all-reduce): total time per step and fraction spent communicating gradients. Single node × 2 GPUs, xl model.
*Deliverable: description of setup + measured time per iteration and time spent communicating gradients.*

**Answer:**

---

## §5.3.1 — `minimal_ddp_flat_benchmarking` (2 pts)

Flatten all gradients into a single all-reduce; compare to per-parameter all-reduce (1 node × 2 GPUs, xl).
*Deliverable: measured time per iteration + communication time for the flattened version, and 1–2 sentences comparing batched vs. individual.*

**Answer:**

---

## §5.3.2 — `ddp_overlap_individual_parameters_benchmarking` (1 pt)

**(a)** Benchmark DDP that overlaps backward compute with per-parameter gradient communication; compare to the per-parameter and flattened variants (1 node × 2 GPUs, xl).
*Deliverable: measured time per iteration + 1–2 sentences comparing the results.*

**Answer:**

**(b)** Profile with Nsight, comparing the initial DDP vs. the overlapped implementation; show that one overlaps compute with communication and the other doesn't.
*Deliverable: 2 screenshots (initial vs. overlapped) showing overlap or lack thereof.*

**Answer:**

---

## §6 — `optimizer_state_sharding_accounting` (5 pts)

**(a)** Peak memory with and without optimizer state sharding (1 node × 2 GPUs, xl): report peak after init, right before, and right after the optimizer step. Do results match expectations? Break memory down by component (params, optimizer states, etc.).
*Deliverable: 2–3 sentence response with peak-memory results and a component breakdown.*

**Answer:**

**(b)** How does sharding affect training speed? Time per iteration with vs. without (1 node × 2 GPUs, xl).
*Deliverable: 2–3 sentence response with timings.*

**Answer:**

**(c)** How does this sharding approach differ from ZeRO stage 1 (ZeRO-DP P_os)?
*Deliverable: 2–3 sentence summary, especially re: memory and communication volume.*

**Answer:**

---

## §7 — `fsdp_accounting` (5 pts)

**(a)** From your §6 analysis, how much peak memory do you expect FSDP to save (ignore all-gather buffers)?
*Deliverable: 2–3 sentence response with your findings.*

**Answer:**

**(b)** Profile xl on 2 GPUs, focusing on the weight all-gather: does communication finish in time for the forward pass?
*Deliverable: 2–3 sentence response with timings + Nsight screenshots.*

**Answer:**

---

## §8.1 — `alternate_ring_all_reduce` (1 pt)

For the alternate ring all-reduce given (W egress bandwidth/device, each x^(i) of size S, N devices), how long does the algorithm take?
*Deliverable: an answer in terms of S, N, W + one-sentence justification.*

**Answer:**

---

## §8.2 — `data_parallel_calcs` (3 pts)

*(Single FFN layer; x:(B,D), W1,W2:(D,D_ff), W3:(D_ff,D); FP16 = 2 bytes; matmul (A,B)(B,C) = 2·A·B·C FLOPs.)*

**(a)** FLOPs for the backward pass with N_DP data parallelism (ignore non-matmul).
*Deliverable: answer in terms of B, D, D_ff, N_DP + one-sentence justification.*

**Answer:**

**(b)** Communication time in the backward pass with N_DP.
*Deliverable: answer in terms of a subset of B, D, D_ff, N_DP, W + one-sentence justification.*

**Answer:**

**(c)** How large can N_DP get before we're communication-bottlenecked?
*Deliverable: an inequality with N_DP on one side, an expression in a subset of B, D, D_ff, C, W on the other + one-sentence justification.*

**Answer:**

---

## §8.3 — `fsdp_calcs` (3 pts)

**(a)** FLOPs for the backward pass with N_FSDP? And the forward pass?
*Deliverable: two answers in terms of B, D, D_ff, N_FSDP + two one-sentence justifications.*

**Answer:**

**(b)** Communication time in the backward pass with N_FSDP? And the forward pass?
*Deliverable: two answers in terms of a subset of B, D, D_ff, N_FSDP, W + two one-sentence justifications.*

**Answer:**

**(c)** How large can N_FSDP get before the backward pass is communication-bottlenecked? The forward pass?
*Deliverable: two inequalities with N_FSDP on one side, expressions in a subset of B, D, D_ff, C, W + two justifications.*

**Answer:**

---

## §8.4 — `tp_calcs` (4 pts)

**(a)** Given dy:(B,D), write the backward pass of the tensor-parallel FFN (W1^(i),W2^(i):(D, D_ff/N_TP); W3^(i):(D_ff/N_TP, D)), producing each device's dW1^(i), dW2^(i), dW3^(i) and the output dx, using communication primitives.
*Deliverable: a series of equations for the backward pass.*

**Answer:**

**(b)** FLOPs for the forward pass with N_TP? And the backward pass?
*Deliverable: two answers in terms of B, D, D_ff, N_TP + two justifications.*

**Answer:**

**(c)** Communication time in the forward pass with N_TP? And the backward pass?
*Deliverable: two answers in terms of a subset of B, D, D_ff, N_TP, W + two justifications.*

**Answer:**

**(d)** How large can N_TP get before the backward pass is communication-bottlenecked? The forward pass?
*Deliverable: two inequalities with N_TP on one side, expressions in a subset of B, D, D_ff, C, W + two justifications.*

**Answer:**

---

## §8.5 — `fsdp_tp_calcs` (6 pts)

**(a)** FLOPs for the forward pass with N_FSDP FSDP + N_TP TP.
*Deliverable: answer in terms of B, D, D_ff, N_FSDP, N_TP + one-sentence justification.*

**Answer:**

**(b)** Communication time in the forward pass with N_FSDP + N_TP (axes overlap; answer is a max of the FSDP and TP costs).
*Deliverable: answer in terms of a subset of B, D, D_ff, N_FSDP, N_TP, W + one-sentence justification.*

**Answer:**

**(c)** Under optimal N_TP, N_FSDP, how large can N = N_TP·N_FSDP get before the forward pass is communication-bottlenecked?
*Deliverable: an inequality with N on one side, an expression in a subset of B, D, D_ff, C, W on the other + a few sentences/equations of justification.*

**Answer:**

**(d)** Same as (c) but the FSDP-axis and TP-axis collectives **cannot** overlap (shared network).
*Deliverable: an inequality with N on one side + a few sentences/equations of justification.*

**Answer:**

---

## §9 — `leaderboard` (10 pts)

Optimize a full fwd+bwd training step with AdamW for an 8B model (beat the 10 s naïve baseline).
*Deliverable: your best wall-clock time for a full training step (submit to the leaderboard).*

**Answer:**
