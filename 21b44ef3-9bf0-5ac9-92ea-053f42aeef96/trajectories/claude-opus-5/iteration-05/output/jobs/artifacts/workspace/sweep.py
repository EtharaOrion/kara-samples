"""Greedy shrink search: from one parent checkpoint, try several single cuts.

Each candidate is a cfg delta applied to the parent's config.  Candidates run
as separate processes (a few at a time -- the models are tiny, the GPU is not
the bottleneck), each warm-started from the parent.

  python sweep.py --parent work/P69.pt --tag s1 --par 4 --steps 12000 \
      --cand '{"logit_scale":false}' --cand '{"self_bias":false}' ...

Prints a table of surviving members per candidate.
"""

import argparse
import copy
import json
import os
import subprocess
import time

import torch

import model_src


def n_params(cfg):
    return model_src.Adder(cfg).n_params()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cand', action='append', default=[])
    ap.add_argument('--par', type=int, default=4)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--E', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.012)
    ap.add_argument('--noise', type=float, default=0.5)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--extra', default='')
    args = ap.parse_args()

    pcfg = torch.load(args.parent, map_location='cpu')['cfg']
    jobs = []
    for i, c in enumerate(args.cand):
        delta = json.loads(c)
        cfg = copy.deepcopy(pcfg)
        cfg.update(delta)
        out = f'work/{args.tag}_{i}.pt'
        log = f'logs/{args.tag}_{i}.log'
        jobs.append(dict(i=i, delta=delta, cfg=cfg, out=out, log=log,
                         n=n_params(cfg)))

    for j in jobs:
        print(f"[{j['i']}] {j['n']:4d} params  {j['delta']}")

    os.makedirs('work', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    running = []
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < args.par:
            j = queue.pop(0)
            cmd = ['python', 'train_ens.py', '--cfg', json.dumps(j['cfg']),
                   '--E', str(args.E), '--steps', str(args.steps),
                   '--lr', str(args.lr), '--out', j['out'],
                   '--init_from', args.parent, '--init_noise', str(args.noise),
                   '--init_top', str(args.top), '--eval_every', '4000']
            if args.extra:
                cmd += args.extra.split()
            j['proc'] = subprocess.Popen(cmd, stdout=open(j['log'], 'w'),
                                         stderr=subprocess.STDOUT)
            print(f"launched [{j['i']}] -> {j['out']}", flush=True)
            running.append(j)
        time.sleep(5)
        for j in list(running):
            if j['proc'].poll() is not None:
                running.remove(j)
                print(f"done [{j['i']}] rc={j['proc'].returncode}", flush=True)

    print('\n=== results ===')
    rows = []
    for j in jobs:
        try:
            ck = torch.load(j['out'], map_location='cpu')
            acc = ck['acc']
            rows.append((float(acc.max()), int((acc >= 0.999).sum()), j))
        except Exception as e:                                   # noqa: BLE001
            rows.append((-1.0, 0, j))
            print(f"[{j['i']}] load failed: {e}")
    rows.sort(key=lambda r: (-r[1], -r[0]))
    for best, n999, j in rows:
        print(f"[{j['i']}] {j['n']:4d}p  best {best:.5f}  n>=0.999 {n999:3d}  "
              f"{j['delta']}")


if __name__ == '__main__':
    main()
