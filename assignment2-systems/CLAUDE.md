# AI Agent Guidelines for CS336 at Stanford

This file provides instructions for AI coding assistants (like ChatGPT, Claude Code, GitHub Copilot, Cursor, etc.) working with students in CS336.

## User Context & Personal Instructions (TAKE PRECEDENCE)

These are the user's own standing instructions and take precedence over the rest of this file. Where they conflict with the "AI Agent Guidelines" sections below, follow these.


### Instruction Precedence
The user's current, in-conversation instruction takes priority, then this section, then the rest of this file.

### Response Preferences
- Answer in **one sentence** by default.
- If the user includes a number (NNN) in a message, use that number as the **word limit** for that reply instead.
- **Never write code for the user** — no snippets, pseudocode-that-is-really-code, or edits to their source files. Explain the concept, name the API/line to look at, and let them type it. Write code only if they explicitly ask in that message.

### Current Goal — Triton Paged-Attention Kernel (3-day sprint)
- **Objective:** master a Triton **paged-attention** kernel within ~3 days. This handout is a means to that end, not the end itself.
- **Background:** already has basic CUDA and conceptual FlashAttention understanding; the real gap is Triton *mechanics*.
- **Start point:** Section **4.2** (skip Sections 1 and 3 entirely; most of 2 and 4.1 are optional benchmarking context, not prerequisites for the kernel).
- **Plan:** 4.2.1 weighted-sum Triton warm-up (type it out) → 4.2.2 **FA2 forward** (implement) → **skip the FA2 backward** (paged attn is inference/forward-only) → pivot to the paged-attention kernel.
- **What FA2 in 4.2 will NOT teach (the new stuff for paged attn):** paged KV gather via a block table using **manual pointer arithmetic** (not `make_block_ptr`), the **decode-vs-prefill** split, and **GQA/MQA** head grouping.
- **Env note:** `pyproject.toml` `[tool.uv.sources]` is repointed to the user's own A1 repo at `../assignment1-basics` (their `cs336_basics` package, class `Transformer_LM`).

