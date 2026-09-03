#!/bin/bash
# memory.sh — memory sweep: xl × precision (fp32 vs bf16), ctx=512, forward only, 2 steps
# torch memory snapshots only; run nsys separately for 2.1.6(a).
set -u

LABEL=xl
CTX=512
MODE=fwd
PRECS=("fp32" "bf16")
BATCH=1
STEPS=2
OUTDIR=results
mkdir -p "$OUTDIR"

for prec in "${PRECS[@]}"; do

  BF16_FLAG=""
  if [ "$prec" = "bf16" ]; then
    BF16_FLAG="--use_bf16"
  fi

  NAME="mem_${LABEL}_${CTX}_${MODE}_${prec}"
  echo "=== $NAME ==="

  PYTHONPATH=. uv run python cssrc/memory_benchmark.py \
    --mode "$MODE" \
    --label "$LABEL" \
    --context_length "$CTX" \
    --batch_size "$BATCH" \
    --steps "$STEPS" \
    --memory_out_name "$OUTDIR/$NAME" \
    --nvtx \
    $BF16_FLAG \
    || echo "!!! FAILED: $NAME (likely OOM) — continuing"

done

echo "done — snapshots in $OUTDIR/"
