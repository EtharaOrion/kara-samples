#!/bin/sh
# End-to-end pipeline that produced /workspace/submission.py.  Everything it
# writes stays under /workspace.  Single GPU; total wall clock ~6 minutes.
set -e
cd /workspace
mkdir -p runs

# 1. train 4096 independent members of the free-form model from random init
python train_parent.py --E 4096 --steps 15000 --places 1,2,3,5,8 --warm_n1 0.2 \
    --ls_loss 4 --eval_every 2500 --eval_B 2048 --out runs/long1.pt

# 2. put the best of them into the shipped gauge (12 free values) and fine-tune
python stage2.py --src runs/long1.pt --topk 16 --rep 32 --steps 3000 \
    --eval_every 500 --out runs/s2a.pt

# 3. collapse the two bank thresholds into one shared one (11 free values)
python stage3.py --src runs/s2a.pt --W1 1.0 --topk 64 --rep 16 --steps 6000 \
    --lr 0.004 --eval_every 1000 --eval_B 8192 --seed 23 --out runs/s3_final.pt

# 4. certify every candidate on the whole domain and emit the graded file
python build.py --src runs/s3_final.pt --out /workspace/submission.py \
    --report runs/chosen_report.json

# 5. independent audit of the graded file, and the constants' band study
python verify.py --path /workspace/submission.py
python band.py --path /workspace/submission.py --out runs/band_study.json
