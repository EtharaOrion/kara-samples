#!/bin/bash
cd /workspace
run () {  # tag d blocks
  python ens_train.py --tag "$1" --d_model "$2" --blocks "$3" \
    --ensemble 192 --batch 1024 --steps 30000 --lr 0.012 \
    --eval_every 2500 --eval_n 100000 --keep_top 6 --seed 100 \
    >> runs/waveA.log 2>&1
  echo "finished $1" >> runs/waveA.log
}
run A_d4_446   4 "1,1,4;1,1,6"
run A_d4_335   4 "1,1,3;1,1,5"
run A_d3_668   3 "1,1,6;1,1,8"
run A_d3_356   3 "1,1,5;1,1,6"
run A_d3_346   3 "1,1,4;1,1,6"
run A_d3_344   3 "1,1,4;1,1,4"
run A_d2_288   2 "1,1,8;1,1,8"
run A_d2_268   2 "1,1,6;1,1,8"
echo "WAVE A DONE" >> runs/waveA.log
