#!/bin/bash
# Evidence that (alpha, kw, lam) are architecture knobs, not carriers of task
# information: take the SAME trained parent, snap the constants to a different
# arbitrary choice, and retrain the 11 parameters there.
set -e
cd /workspace
A=$1; K=$2; L=$3; TAG=$4
python reduce.py --ckpt runs/cold_v_s1.pt --out runs/red_$TAG.pt --keep 8 \
  --alpha $A --kw $K --lam $L
python - <<PY
import torch
ck = torch.load('runs/red_$TAG.pt', map_location='cpu')
k = max(1, int((ck['acc'] >= 0.99).sum()))
ck['params'] = {a: v[:k].clone() for a, v in ck['params'].items()}
ck['acc'] = ck['acc'][:k]
torch.save(ck, 'runs/red_${TAG}_top.pt')
print('warm members', k)
PY
python train.py --E 256 --steps 12000 --lr 0.004 --seed 61 --sigma 0.06 \
  --init runs/red_${TAG}_top.pt --keep 16 --out runs/ft_$TAG.pt \
  --cfg "{\"code_fix\":1,\"free_rb\":false,\"free_alpha\":false,\"free_e1\":false,\"free_ls\":true,\"free_kw\":false,\"free_lam\":false,\"alpha\":$A,\"kw\":$K,\"lam\":$L}"
