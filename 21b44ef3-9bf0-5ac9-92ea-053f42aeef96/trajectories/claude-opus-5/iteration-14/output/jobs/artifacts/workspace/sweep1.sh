set -e
run () {  # name C U extra...
  name=$1; shift; C=$1; shift; U=$1; shift
  python train.py --E 512 --batch 512 --steps 3000 --eval_every 500 --C $C --U $U \
      --out ckpt/$name.pt "$@" > logs/$name.log 2>&1
  echo "$name : $(grep -o 'best [0-9.]*' logs/$name.log | tail -1)  $(tail -1 logs/$name.log | cut -c1-90)"
}
run c1u4      1 4 --seed 1
run c1u6      1 6 --seed 2
run c2u2      2 2 --seed 3
run c2u4      2 4 --seed 4
run c1u4_warm 1 4 --seed 5 --warm_places 1,2 --warm_frac 0.25
run c2u4_warm 2 4 --seed 6 --warm_places 1,2 --warm_frac 0.25
