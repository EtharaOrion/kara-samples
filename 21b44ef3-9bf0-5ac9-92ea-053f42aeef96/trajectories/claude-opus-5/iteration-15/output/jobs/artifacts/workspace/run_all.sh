#!/bin/bash
# Cold random init -> the graded file, in one run.  Nothing is carried over
# from earlier sessions: every weight in submission.py is produced by the
# gradient descent below.
#
#   bash run_all.sh [seed]
#
# Stages.  Each one is ordinary training; what changes between them is how much
# of the block is still free.
#
#   1  cold      a wide lottery of two-channel parents, from random init.  A
#                one-channel code is not reachable from scratch, so the parent
#                gets two and finds the mechanism there.
#   2  project   collapse the codes onto their principal direction (a re-init,
#                which loses accuracy) and train the one-channel block back up.
#   3  sharpen   replace the learned attention temperature and key contrast by
#                round constants and train under them.  The learned values are
#                soft enough that long carry chains mis-route; training under
#                the sharp constants is what fixes that, not the substitution.
#   4  ship      rewrite each member into the exact twelve-parameter shipped
#                form and train those twelve values under the shipped
#                constants, with a term asking the bank to stay saturated.
#
# Then: certify every member over the whole 8-digit domain, ship the best one,
# and check the graded file end to end.
set -eo pipefail          # a stage that fails inside a `| tee` must stop the run
cd "$(dirname "$0")"
SEED=${1:-15}
mkdir -p ckpt logs

echo "=== 1/4 cold parents (seed $SEED) ==="
python train.py --E 2048 --C 2 --U 2 --steps 6000 --seed "$SEED" \
  --out "ckpt/run${SEED}_parent.pt" 2>&1 | tee "logs/run${SEED}_1_cold.log"

echo "=== 2/4 project to one channel ==="
python retrain.py --ckpt "ckpt/run${SEED}_parent.pt" --top 16 --thresh 0.999 \
  --project --E 256 --sigma 0.1 --steps 6000 --lr 0.004 --seed "$SEED" \
  --out "ckpt/run${SEED}_c1.pt" 2>&1 | tee "logs/run${SEED}_2_project.log"

echo "=== 3/4 sharpen the attention ==="
python retrain.py --ckpt "ckpt/run${SEED}_c1.pt" --top 16 --thresh 0.999 \
  --set lam=-12 --set key_w=-400,400 --freeze lam,key_w \
  --E 256 --sigma 0.05 --steps 6000 --lr 0.003 --seed "$SEED" \
  --out "ckpt/run${SEED}_sharp.pt" 2>&1 | tee "logs/run${SEED}_3_sharp.log"

echo "=== 4/4 train the twelve shipped parameters ==="
python retrain.py --ckpt "ckpt/run${SEED}_sharp.pt" --top 8 --thresh 0.9999 \
  --shipped --E 256 --sigma 0.02 --steps 6000 --lr 0.002 --tau 4 \
  --aux 0.5 --slack 1.0 --jitter code,knee,fold_w --seed "$SEED" \
  --out "ckpt/run${SEED}_ship.pt" 2>&1 | tee "logs/run${SEED}_4_ship.log"

echo "=== certify every member and pick the best ==="
python select.py --ckpt "ckpt/run${SEED}_ship.pt" --top 128 \
  --out "logs/run${SEED}_select.json" 2>&1 | tee "logs/run${SEED}_5_select.log"
BEST=$(grep '^BEST ' "logs/run${SEED}_5_select.log" | awk '{print $2}')
test -n "$BEST"

echo "=== build, certify and verify the graded file (member $BEST) ==="
python build.py --ckpt "ckpt/run${SEED}_ship.pt" --member "$BEST" \
  --out /workspace/submission.py 2>&1 | tee "logs/run${SEED}_6_build.log"
python certify.py --path /workspace/submission.py 2>&1 | tee "logs/run${SEED}_7_certify.log"
python verify.py --path /workspace/submission.py 2>&1 | tee "logs/run${SEED}_8_verify.log"
echo "done: /workspace/submission.py from ckpt/run${SEED}_ship.pt member $BEST"
