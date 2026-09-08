#!/bin/bash
# Full pipeline, from random initialisation to the graded file.
set -e
mkdir -p /workspace/logs /workspace/ckpt
cd /workspace

python stage1.py --E 65536 --steps 3000 --seed 11 --keep 512 \
    --out ckpt/stage1.pt 2>&1 | tee logs/stage1.log

python stage2.py --parents ckpt/stage1.pt --reps 24 --steps 6000 --batch 192 \
    --seed 12 --keep 1024 --out ckpt/stage2.pt 2>&1 | tee logs/stage2.log

python stage3.py --src ckpt/stage2.pt --reps 8 --steps 4000 --seed 13 \
    --out ckpt/stage3.pt 2>&1 | tee logs/stage3.log

echo "pipeline done"
