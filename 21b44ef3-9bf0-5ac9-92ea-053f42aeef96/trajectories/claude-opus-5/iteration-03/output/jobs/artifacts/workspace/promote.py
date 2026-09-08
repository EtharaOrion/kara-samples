"""Turn a finished training run into a candidate submission file.

Four steps, in order:
  1. pick the best member of the checkpoint (by the run's own large-sample
     rescoring, recorded in the sibling .json),
  2. drop `logit_scale` -- one positive scalar multiplying every logit, which
     rescales the softmax temperature but provably cannot move the arg max, so
     the smaller model's predictions are checked to be bit-identical,
  3. gauge-fix the factorised embedding, which removes `emb_up` using the O(d)
     symmetry of the residual stream (see gauge.py); also checked exactly,
  4. emit cand_<n_params>.py and run the full verification suite on it.

Steps 2 and 3 are function-preserving rewrites, not retraining: each one
refuses to write its output unless the smaller model reproduces the larger
one's predictions on a large fresh sample.

Nothing is copied to submission.py automatically; that stays a manual decision
made after reading the verification output.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

from drop_scale import strip
from gauge import fix as gauge_fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="run tag, i.e. ckpt/<tag>.pt")
    ap.add_argument("--member", type=int, default=None,
                    help="default: best member per ckpt/<tag>.json")
    ap.add_argument("--n", type=int, default=2_000_000,
                    help="uniform held-out pairs for verification")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--keep_scale", action="store_true",
                    help="skip the logit_scale strip")
    ap.add_argument("--keep_emb_up", action="store_true",
                    help="skip the embedding gauge fix")
    args = ap.parse_args()

    ckpt = f"/workspace/ckpt/{args.tag}.pt"
    member = args.member
    if member is None:
        meta = json.load(open(f"/workspace/ckpt/{args.tag}.json"))
        member = meta["top"][0]["member"]
        print(f"[promote] best member {member} "
              f"(acc {meta['top'][0]['acc']:.6f} on the run's own eval mix)")

    src, mem = ckpt, member
    if not args.keep_scale:
        src = f"/workspace/ckpt/{args.tag}_nls.pt"
        strip(ckpt, src, mem, n_check=200_000)
        mem = 0

    cfg = torch.load(src, map_location="cpu", weights_only=False)["cfg"]
    if cfg.get("emb_rank") and not cfg.get("emb_fixed_up") \
            and not args.keep_emb_up:
        nxt = f"/workspace/ckpt/{args.tag}_fu.pt"
        gauge_fix(src, nxt, mem, n_check=200_000)
        src, mem = nxt, 0

    n = sum(v.numel() for v in
            torch.load(src, map_location="cpu",
                       weights_only=False)["members"][mem].values())
    out = f"/workspace/cand_{n}.py"
    for cmd in ([sys.executable, "build_submission.py", src,
                 "--member", str(mem), "--out", out],
                [sys.executable, "verify.py", "--path", out,
                 "--n", str(args.n), "--reps", str(args.reps),
                 "--api_n", "3000"]):
        print(f"[promote] $ {' '.join(os.path.basename(c) for c in cmd[1:2])} ...",
              flush=True)
        subprocess.run(cmd, check=True, cwd="/workspace")
    print(f"[promote] candidate ready at {out} ({n} parameters)")


if __name__ == "__main__":
    main()
