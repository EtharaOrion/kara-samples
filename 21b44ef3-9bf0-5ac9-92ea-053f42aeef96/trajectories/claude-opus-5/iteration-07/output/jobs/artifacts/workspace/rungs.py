"""Run a batch of candidate architecture cuts from one parent checkpoint.

Each candidate is a cfg delta; every candidate is warm-started from the parent
and then retrained.  Prints a table of (params, best held-out accuracy).
"""
import argparse, json, os, subprocess, sys, time
import torch

from model_src import DigitPairAdder, default_cfg, n_params


def count(cfg):
    return n_params(DigitPairAdder(cfg))


def applicable(cfg, delta):
    c = dict(cfg); c.update(delta)
    if c["f1_tie"] and c["f1_out"] != "axis":
        return False
    if c["f2_tie"] and (c["f2_out"] != "axis" or c["U2"] % 2):
        return False
    if c["wo"] == "fix" and c["kv"] != "id":
        return False
    if c["code_fix"] > 0 and c["C"] != 1:
        return False
    if c["KV"] >= c["D"] or c["C"] > c["D"]:
        return False
    if c["U1"] < 1 or c["U2"] < 1:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent")
    ap.add_argument("--cands", required=True, help="JSON list of [name, delta]")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=0.008)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--places", default="8,14")
    ap.add_argument("--sched", default="cos")
    ap.add_argument("--dir", default="runs")
    ap.add_argument("--tag", default="r")
    args = ap.parse_args()

    ck = torch.load(args.parent, map_location="cpu", weights_only=False)
    pcfg = dict(default_cfg()); pcfg.update(ck["cfg"])
    p0 = count(pcfg)
    cands = json.loads(args.cands)
    jobs = []
    for name, delta in cands:
        if not applicable(pcfg, delta):
            print(f"skip {name}: not applicable")
            continue
        c = dict(pcfg); c.update(delta)
        jobs.append((name, delta, count(c)))
    jobs.sort(key=lambda j: j[2])
    print(f"parent {args.parent}  params={p0}  acc={ck.get('acc')}")
    for n, d, p in jobs:
        print(f"  candidate {n:22s} -> {p} params ({p0 - p} saved)")

    running, results = [], []
    os.makedirs(args.dir, exist_ok=True)
    while jobs or running:
        while jobs and len(running) < args.par:
            name, delta, p = jobs.pop(0)
            out = os.path.join(args.dir, f"{args.tag}_{name}.pt")
            log = os.path.join("logs", f"{args.tag}_{name}.log")
            cmd = [sys.executable, "train_ens.py", "--init", args.parent,
                   "--cfg", json.dumps(delta), "--E", str(args.E),
                   "--steps", str(args.steps), "--sigma", str(args.sigma),
                   "--lr", str(args.lr), "--seed", str(args.seed),
                   "--places", str(args.places), "--sched", args.sched,
                   "--eval_every", "500", "--eval_n", "16384", "--out", out]
            f = open(log, "w")
            running.append((name, p, out, subprocess.Popen(cmd, stdout=f,
                                                          stderr=subprocess.STDOUT), f))
            print(f"launched {name}", flush=True)
        time.sleep(5)
        for r in list(running):
            if r[3].poll() is not None:
                r[4].close()
                running.remove(r)
                try:
                    c = torch.load(r[2], map_location="cpu", weights_only=False)
                    acc = float(c["acc"])
                except Exception as e:
                    acc = float("nan")
                results.append((r[0], r[1], acc, r[2]))
                print(f"done {r[0]:22s} params={r[1]:4d} acc={acc:.5f}", flush=True)

    print("\n==== summary (parent %d params) ====" % p0)
    for name, p, acc, path in sorted(results, key=lambda x: (-x[2], x[1])):
        print(f"  {name:22s} params={p:4d}  acc={acc:.6f}  {path}")


if __name__ == "__main__":
    main()
