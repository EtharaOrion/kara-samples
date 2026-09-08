"""Run several training configs concurrently on one GPU and report the results.

A job is ``(tag, cfg)`` or ``(tag, cfg, extra)`` where ``extra`` is a list of
additional argv for train_ens.py (per-job ``--init_from``, ``--lr``, ...).
Later flags win, so per-job extras override the scan-wide defaults.
"""
import json, subprocess, sys, time


def run(jobs, E, steps, batch=1024, lr=0.012, par=4, extra=()):
    procs, out = [], []
    for job in jobs:
        tag, cfg = job[0], job[1]
        per = list(job[2]) if len(job) > 2 else []
        cmd = [sys.executable, "train_ens.py", "--cfg", json.dumps(cfg), "--E", str(E),
               "--steps", str(steps), "--batch", str(batch), "--lr", str(lr),
               "--out", f"/workspace/ck/{tag}.pt", "--tag", tag,
               "--eval_every", str(max(1000, steps // 8))] + list(extra) + per
        out.append(open(f"/workspace/logs/{tag}.log", "w"))
        procs.append((tag, subprocess.Popen(cmd, stdout=out[-1], stderr=subprocess.STDOUT)))
        while sum(p.poll() is None for _, p in procs) >= par:
            time.sleep(3)
    for _, p in procs:
        p.wait()
    for f in out:
        f.close()
    res = []
    for job in jobs:
        tag = job[0]
        last = [l for l in open(f"/workspace/logs/{tag}.log") if "DONE" in l]
        res.append((tag, last[-1].strip() if last else "FAILED"))
    return res
