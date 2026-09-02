#!/bin/bash
# memory.sh — memory sweep: xl × context length × mode, mixed precision (bf16)
# torch memory snapshots only; run nsys separately for 2.1.6(a).
set -u

LABEL=xl
CTXS=(128 2048)
MODES=("fwd" "fwd_bwd" "fwd_bwd_opt")
STEPS=3
OUTDIR=results
mkdir -p "$OUTDIR"

for ctx in "${CTXS[@]}"; do
  for mode in "${MODES[@]}"; do

    # xl at ctx 2048 does not fit in 80GB with batch 4
    if [ "$ctx" -ge 2048 ]; then
      BATCH=1
    else
      BATCH=4
    fi

    NAME="mem_${LABEL}_${ctx}_${mode}_bf16"
    echo "=== $NAME ==="

    PYTHONPATH=. uv run python cssrc/memory_benchmark.py \
      --mode "$mode" \
      --label "$LABEL" \
      --context_length "$ctx" \
      --batch_size "$BATCH" \
      --steps "$STEPS" \
      --memory_out_name "$OUTDIR/$NAME" \
      --nvtx \
      --use_bf16 \
      || echo "!!! FAILED: $NAME (likely OOM) — continuing"

  done
done

echo "done — snapshots in $OUTDIR/"
