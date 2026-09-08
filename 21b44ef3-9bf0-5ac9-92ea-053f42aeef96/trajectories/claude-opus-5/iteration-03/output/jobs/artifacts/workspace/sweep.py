"""Run a queue of train.py configs, N at a time, and print a frontier summary."""
import argparse
import json
import os
import subprocess
import sys
import time

from ens import single_param_count

CKPT = "/workspace/ckpt"
LOGS = "/workspace/logs"


def cfg_tag(c):
    t = f"d{c['d']}_{c['f1']}_{c['f2']}"
    if c.get("dv"):
        t += f"_v{c['dv']}"
    if c.get("no_res_bias"):
        t += "_nb"
    if c.get("share_qk"):
        t += "_sq"
    if c.get("emb_rank"):
        t += f"_er{c['emb_rank']}"
    for f, s in (("no_norm", "nn"), ("no_qb", "nq"), ("no_sb", "ns"),
                 ("no_ls", "nl"), ("no_bin2", "nbi"),
                 ("no_bout2", "nbo"), ("no_alibi", "na"),
                 ("emb0_zero", "e0"),
                 ("emb_fixed_up", "fu"),
                 ("plane_io", "pio")):
        if c.get(f):
            t += "_" + s
    for k in ("steps", "lr", "act", "E", "rounds", "seed"):
        if k in c:
            t += f"_{k}{c[k]}"
    return t


def params(c):
    return single_param_count(c["d"], c["f1"], c["f2"], d_v=c.get("dv", 0),
                              res_bias=not c.get("no_res_bias", False),
                              share_qk=c.get("share_qk", False),
                              q_bias=not c.get("no_qb", False),
                              self_bias=not c.get("no_sb", False),
                              out_scale=not c.get("no_ls", False),
                              res_bias_in=(not c.get("no_res_bias", False)
                                           and not c.get("no_bin2", False)),
                              res_bias_out=(not c.get("no_res_bias", False)
                                            and not c.get("no_bout2", False)),
                              emb_rank=c.get("emb_rank", 0),
                              alibi=not c.get("no_alibi", False),
                              emb0_zero=c.get("emb0_zero", False),
                              emb_fixed_up=c.get("emb_fixed_up", False),
                              plane_io=c.get("plane_io", False))


def build_cmd(c, tag):
    cmd = [sys.executable, "train.py", "--d", str(c["d"]), "--f1", str(c["f1"]),
           "--f2", str(c["f2"]), "--tag", tag]
    for k in ("E", "steps", "batch", "lr", "seed", "dv", "act", "rounds",
              "clip", "warm", "final_eval", "reinit_frac", "wd", "init_from",
              "init_member", "init_sigma", "alibi_mean", "emb_rank"):
        if k in c:
            cmd += [f"--{k}", str(c[k])]
    if c.get("no_res_bias"):
        cmd += ["--no_res_bias"]
    if c.get("share_qk"):
        cmd += ["--share_qk"]
    for f in ("no_norm", "no_qb", "no_sb", "no_ls", "no_bin2", "no_bout2", "no_alibi", "emb0_zero", "emb_fixed_up", "plane_io"):
        if c.get(f):
            cmd += [f"--{f}"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--no_sort", action="store_true")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    queue = json.load(open(args.queue))
    if not args.no_sort:
        queue.sort(key=params)

    running = []
    pending = list(queue)
    while pending or running:
        while pending and len(running) < args.par:
            c = pending.pop(0)
            tag = c.get("tag") or cfg_tag(c)
            if os.path.exists(os.path.join(CKPT, tag + ".json")):
                print(f"skip {tag} (done)", flush=True)
                continue
            lf = open(os.path.join(LOGS, tag + ".log"), "w")
            p = subprocess.Popen(build_cmd(c, tag), stdout=lf, stderr=lf,
                                 cwd="/workspace")
            print(f"start {tag}  ({params(c)} params)", flush=True)
            running.append((tag, p, lf, time.time()))
        time.sleep(5)
        for r in list(running):
            tag, p, lf, t0 = r
            if p.poll() is not None:
                lf.close()
                running.remove(r)
                j = os.path.join(CKPT, tag + ".json")
                if os.path.exists(j):
                    d = json.load(open(j))
                    print(f"done  {tag}  params={d['params']} "
                          f"best={d['top'][0]['acc']:.6f} "
                          f"n>=99.99%={d['n_ge_9999']} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                else:
                    print(f"FAIL  {tag} rc={p.returncode} "
                          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
