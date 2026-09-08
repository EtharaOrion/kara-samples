#!/bin/sh
# Independent-seed reproducibility check: re-run the whole pipeline from a cold
# phase-1 lottery seeded differently from the runs that produced the shipped
# weights, and see whether it lands on the same solution.
set -e
cd /workspace
while pgrep -f "train.py --ensemble 131072" > /dev/null; do sleep 20; done
python scorer.py runs/big1.pt
python pool.py --out runs/parents1.pt runs/big1.pt
python train.py --init_from runs/parents1.pt --ensemble 32768 --steps 6000 \
  --lr 0.004 --pct_start 0.1 --metric l1 --ls 4.0 --norm 0 --batch 128 \
  --jitter 0.01 --places 2,3,5,8 --eval_places 8,12 --eval_batch 2048 \
  --eval_every 1000 --seed 23 --out phase2b.pt
python select.py --ckpt runs/phase2b.pt --top 8
