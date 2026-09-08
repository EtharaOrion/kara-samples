#!/bin/bash
cd /workspace
run () {  # tag d m1 m2
  python ens_train.py --tag "$1" --d_model "$2" --blocks "0,0,$3;1,1,$4" \
    --rel_mode ramp --ensemble 256 --batch 1024 --steps 30000 --lr 0.012 \
    --eval_every 2500 --eval_n 100000 --keep_top 6 --seed 200 \
    >> runs/waveB.log 2>&1
  echo "finished $1" >> runs/waveB.log
}
run B_d4_m4_6   4 4 6
run B_d3_m6_6   3 6 6
run B_d3_m5_5   3 5 5
run B_d3_m4_5   3 4 5
run B_d3_m4_4   3 4 4
run B_d2_m8_6   2 8 6
run B_d3_m3_4   3 3 4
run B_d2_m6_6   2 6 6
run B_d2_m5_5   2 5 5
run B_d4_m3_5   4 3 5
echo "WAVE B DONE" >> runs/waveB.log
