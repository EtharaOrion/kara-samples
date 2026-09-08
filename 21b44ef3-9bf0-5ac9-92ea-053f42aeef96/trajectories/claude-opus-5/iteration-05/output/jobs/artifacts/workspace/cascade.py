"""Walk a ladder of ever-smaller configs, warm-starting each rung from the last.

Each rung applies a cfg delta to the running config and trains several variants
(different init noise / lr) in parallel; the best checkpoint becomes the parent
of the next rung.  A rung that reaches --thresh is kept, and if it is smaller
than whatever is currently at the graded path the submission is rewritten.

  python cascade.py --parent work/s1_0.pt --ladder ladder.json --tag c1
"""

import argparse
import copy
import json
import shutil
import subprocess
import time

import torch

import model_src

VARIANTS = [(0.10, 0.008), (0.25, 0.012), (0.50, 0.012)]


SUB = '/workspace/submission.py'
BAK = 'work/submission_prev.py'


def n_params(cfg):
    return model_src.Adder(cfg).n_params()


def shipped_params():
    """Parameter count of whatever currently sits at the graded path."""
    try:
        import check_submission
        model, _ = check_submission.load(SUB).build_model()
        return sum(p.numel() for p in model.parameters())
    except Exception:                                            # noqa: BLE001
        return 10 ** 9


def ship(path, tag):
    """Write the submission, smoke-test it, and roll back if it is broken."""
    shutil.copyfile(SUB, BAK)
    r = subprocess.run(['python', 'build_submission.py', '--ckpt', path,
                        '--note', f'cascade rung {tag}'], capture_output=True, text=True)
    print('  ship:', (r.stdout + r.stderr).strip().replace('\n', ' | '), flush=True)
    if r.returncode != 0:
        shutil.copyfile(BAK, SUB)
        return False
    c = subprocess.run(['python', 'check_submission.py', '--n', '400'],
                       capture_output=True, text=True)
    print('  check:', (c.stdout + c.stderr).strip().replace('\n', ' | ')[:300], flush=True)
    if c.returncode != 0:
        print('  ship FAILED its smoke test, rolling back', flush=True)
        shutil.copyfile(BAK, SUB)
        return False
    return True


def run_rung(cfg, parent, tag, steps, E, top, par_variants, extra=''):
    jobs = []
    for vi, (noise, lr) in enumerate(par_variants):
        out = f'work/{tag}_v{vi}.pt'
        log = f'logs/{tag}_v{vi}.log'
        cmd = ['python', 'train_ens.py', '--cfg', json.dumps(cfg),
               '--E', str(E), '--steps', str(steps), '--lr', str(lr),
               '--out', out, '--init_from', parent, '--init_noise', str(noise),
               '--init_top', str(top), '--eval_every', str(max(2000, steps // 4))]
        if extra:
            cmd += extra.split()
        p = subprocess.Popen(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
        jobs.append(dict(vi=vi, out=out, log=log, proc=p, noise=noise, lr=lr))
    for j in jobs:
        j['proc'].wait()
    best = None
    for j in jobs:
        try:
            ck = torch.load(j['out'], map_location='cpu')
        except Exception as e:                                   # noqa: BLE001
            print(f"  v{j['vi']} failed to load: {e}", flush=True)
            continue
        acc = ck['acc']
        n999 = int((acc >= 0.999).sum())
        print(f"  v{j['vi']} noise {j['noise']} lr {j['lr']}: "
              f"best {float(acc.max()):.5f}  n>=0.999 {n999}", flush=True)
        key = (n999, float(acc.max()))
        if best is None or key > best[0]:
            best = (key, j['out'])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', required=True)
    ap.add_argument('--ladder', required=True)
    ap.add_argument('--tag', default='c')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--E', type=int, default=256)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--thresh', type=float, default=0.999)
    ap.add_argument('--ship', action='store_true',
                    help='rewrite /workspace/submission.py on every good rung')
    args = ap.parse_args()

    ladder = json.load(open(args.ladder))
    cfg = model_src.default_cfg(**torch.load(args.parent, map_location='cpu')['cfg'])
    parent = args.parent
    best_shipped = min(n_params(cfg), shipped_params())
    print(f'start {parent}  {n_params(cfg)} params, '
          f'graded path holds {shipped_params()}', flush=True)

    for ri, rung in enumerate(ladder):
        cfg = copy.deepcopy(cfg)
        cfg.update(rung['delta'])
        cfg = model_src.default_cfg(**cfg)
        n = n_params(cfg)
        tag = f"{args.tag}{ri}_{rung.get('name', 'r')}"
        steps = rung.get('steps', args.steps)
        var = rung.get('variants')
        var = VARIANTS if var is None else [tuple(v) for v in var]
        print(f"\n=== rung {ri} {rung.get('name','')}: {n} params  "
              f"delta {rung['delta']}  steps {steps}", flush=True)
        t0 = time.time()
        best = run_rung(cfg, parent, tag, steps, rung.get('E', args.E),
                        args.top, var, rung.get('extra', ''))
        if best is None:
            print('  no checkpoint produced, stopping', flush=True)
            break
        (n999, acc), path = best
        print(f'  -> {path}  best {acc:.5f}  n>=0.999 {n999}  '
              f'({time.time()-t0:.0f}s)', flush=True)
        if acc < args.thresh:
            print('  rung failed the accuracy bar, stopping', flush=True)
            break
        parent = path
        # a rung may have annealed something into the config (norm skips)
        cfg = model_src.default_cfg(**torch.load(path, map_location='cpu')['cfg'])
        if args.ship and n < best_shipped and ship(path, tag):
            best_shipped = n
    print(f'\ncascade done, last good parent {parent}', flush=True)


if __name__ == '__main__':
    main()
