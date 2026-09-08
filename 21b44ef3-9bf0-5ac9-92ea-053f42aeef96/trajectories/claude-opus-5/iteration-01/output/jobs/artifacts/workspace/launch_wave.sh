#!/bin/bash
# launch_wave.sh <steps> <batch> <lr> "<tag> <d_model> <layers> <extra flags> <seed>" ...
STEPS=$1; BATCH=$2; LR=$3; shift 3
cd /workspace
for spec in "$@"; do
  set -- $spec
  TAG=$1; DM=$2; LAYERS=$3; SEED=${4:-0}; shift 4 2>/dev/null || shift $#
  EXTRA="$*"
  OMP_NUM_THREADS=1 nohup python train.py \
      --steps "$STEPS" --batch "$BATCH" --lr "$LR" \
      --eval-every 10000 --eval-n 50000 --q-bias --learn-scale \
      --seed "$SEED" --d-model "$DM" --layers "$LAYERS" $EXTRA \
      --out "runs/$TAG.pt" > "runs/$TAG.log" 2>&1 &
  echo "launched $TAG (d=$DM layers=$LAYERS seed=$SEED extra='$EXTRA')"
done
