"""Focused phase-1 comparison, scored by usable parents (linear code)."""
import subprocess, sys, torch, scorer

CFGS = [
    dict(tag="sig0.05", code_sigma=0.05),
    dict(tag="sig0.3",  code_sigma=0.3),
    dict(tag="sig1",    code_sigma=1.0),
    dict(tag="sig3",    code_sigma=3.0),
    dict(tag="sig0.3r", code_sigma=0.3, digit_ramp=0.3),
    dict(tag="sig3r",   code_sigma=3.0, digit_ramp=0.3),
]
for cfg in CFGS:
    tag = "s3_" + cfg.pop("tag")
    cmd = [sys.executable, "train.py", "--ensemble", "4096", "--steps", "25000",
           "--lr", "0.02", "--metric", "l1", "--ls", "1.0", "--norm", "0",
           "--batch", "256", "--exhaustive1", "0", "--places", "1",
           "--eval_places", "1", "--eval_every", "5000", "--out", tag + ".pt"]
    for k, v in cfg.items():
        cmd += ["--" + k, str(v)]
    subprocess.run(cmd, capture_output=True, text=True)
    p = {k: v.cuda() for k, v in torch.load("runs/%s.pt" % tag)["params"].items()}
    res, step = scorer.residual(scorer.code_table(p))
    k1 = p["knee"][:, 1] / step
    good = res < 0.08
    print("%-12s linear %4d   linear&knee %4d   best_res %.4f" %
          (tag, int(good.sum()), int((good & (k1 > 9) & (k1 < 10)).sum()), float(res.min())),
          flush=True)
