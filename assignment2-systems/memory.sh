#!/bin/bash
# memory.sh — memory sweep: xl × context length × mode, mixed precision (bf16)
set -u

LABEL=xl
CTXS=(128 2048)
MODES=("fwd" "fwd_bwd" "fwd_bwd_opt")
OUTDIR=results
mkdir -p "$OUTDIR"

for ctx in "${CTXS[@]}"; do
  for mode in "${MODES[@]}"; do

    NAME="mem_${LABEL}_${ctx}_${mode}_bf16"
    echo "=== $NAME ==="

    PYTHONPATH=. nsys profile \
      -t cuda,nvtx,cublas \
      -c cudaProfilerApi \
      --capture-range-end=stop \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      -o "$OUTDIR/$NAME" \
      uv run python cssrc/nsys_benchmark.py \
        --mode "$mode" \
        --label "$LABEL" \
        --context_length "$ctx" \
        --nvtx \
        --use_bf16 \
      || echo "!!! FAILED: $NAME (likely OOM) — continuing"

  done
done

echo "done — reports in $OUTDIR/"
