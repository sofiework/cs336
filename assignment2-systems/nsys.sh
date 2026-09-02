#!/bin/bash
# nsys.sh — sweep model size × precision × mode
set -u

MODES=("fwd" "fwd_bwd")
LABELS=("small" "medium" "large" "xl")
CTX=512
OUTDIR=results
mkdir -p "$OUTDIR"

for label in "${LABELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for prec in fp32 bf16; do

      if [ "$prec" = "bf16" ]; then
        BF16_FLAG="--use_bf16"
      else
        BF16_FLAG=""
      fi

      NAME="nsys_${label}_${CTX}_${mode}_${prec}"
      echo "=== $NAME ==="

      PYTHONPATH=. nsys profile \
        -t cuda,nvtx,cublas \
        -c cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        -o "$OUTDIR/$NAME" \
        uv run python cssrc/nsys_benchmark.py \
          --mode "$mode" \
          --label "$label" \
          --nvtx \
          $BF16_FLAG \
        || echo "!!! FAILED: $NAME (likely OOM) — continuing"

    done
  done
done

echo "done — reports in $OUTDIR/"