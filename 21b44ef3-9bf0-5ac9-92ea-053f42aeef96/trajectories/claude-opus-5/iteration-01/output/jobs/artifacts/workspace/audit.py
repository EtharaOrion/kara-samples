"""Score every checkpoint against the acceptance gates, straight from the .pt.

Same measurements verify.py makes on a built submission, but cheap enough to
sweep the whole runs/ directory: exact match on held-out uniform pairs and on
maximal carry chains, then whether the last attention layer is a real
content-addressed look-up (row entropy, argmax agreement with the carry target,
carry read-out) and what happens to the chains when its pattern is frozen.
"""
import argparse, glob, os, sys

import torch

import data as D
import model_src
from model_src import AdderTransformer
from verify import chain_cases, to_tokens, exact_match


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(d_model=ck["d_model"], layers=ck["layers"],
               pos_mode=ck.get("pos_mode", "rank1"), tie_head=ck.get("tie_head", True),
               head_bias=ck.get("head_bias", True), q_bias=ck.get("q_bias", False),
               learn_scale=ck.get("learn_scale", False))
    m = AdderTransformer(**cfg)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def audit(path, dev, ho, ch):
    m, ck = load(path)
    m = m.to(dev)
    n = sum(p.numel() for p in m.parameters())
    acc = exact_match(m, *ho)
    cacc = exact_match(m, *ch)

    model_src.STORE_ATTN = True
    layers = [b.attn for b in m.blocks if b.attn is not None]
    a, b = ch[0][:8192], ch[1][:8192]
    _ = m(*to_tokens(a, b))

    s = (a + b).long()
    s = torch.cat([torch.zeros_like(s[:, :1]), s, torch.zeros_like(s[:, :1])], 1)
    gen, tr = s >= 10, s == 9
    carry = torch.zeros_like(s, dtype=torch.bool)
    cur = torch.zeros(s.shape[0], dtype=torch.bool, device=dev)
    for i in range(10):
        carry[:, i] = cur
        cur = gen[:, i] | (tr[:, i] & cur)
    nonprop = ~tr

    att = layers[-1].attn
    ent = float((-(att[:, :, 1:, :] * att[:, :, 1:, :].clamp_min(1e-9).log()).sum(-1)).mean())
    hit = ag = near = tot = ntot = 0
    for i in range(1, 10):
        mass = (att[:, :, i, :i].mean(1) * gen[:, :i].float()).sum(-1)
        hit += int(((mass > 0.5) == carry[:, i]).sum())
        j = torch.arange(i, device=dev)
        target = (nonprop[:, :i].float() * (j + 1).float()).argmax(-1)
        need = carry[:, i]        # only these rows pin down where the head must look
        pick = att[:, :, i, :i].mean(1).argmax(-1)
        near += int(((pick == target) & need).sum())
        ag += int(gen[:, :i].gather(1, pick[:, None])[need].sum())
        ntot += int(need.sum())
        tot += a.shape[0]

    # freeze every attention layer to the mean pattern it produces on the very
    # data it is scored on, then re-run the chains
    _ = m(*to_tokens(ch[0][:8192], ch[1][:8192]))
    for l in layers:
        l.frozen = l.attn.mean(0, keepdim=True)
    orig = type(layers[0]).forward

    def frozen_forward(self, x, mask):
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        v = self.wv(x).view(B, T, H, Dh).transpose(1, 2)
        y = torch.matmul(self.frozen.to(x.dtype).expand(B, H, T, T), v)
        return self.wo(y.transpose(1, 2).reshape(B, T, H * Dh))

    type(layers[0]).forward = frozen_forward
    fro = exact_match(m, *ch)
    type(layers[0]).forward = orig
    model_src.STORE_ATTN = False

    ok = acc >= 0.99 and cacc >= 0.99 and ag / max(ntot, 1) >= 0.95 and hit / tot >= 0.99 and fro < 0.50
    return dict(path=os.path.basename(path), params=n, acc=acc, chain=cacc, ent=ent,
                argmax=ag / max(ntot, 1), nearest=near / max(ntot, 1), readout=hit / tot, frozen=fro, ok=ok,
                cfg="d=%d %s %s" % (ck["d_model"], ck["cfg"]["layers"], ck.get("pos_mode", "rank1")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", default=sorted(glob.glob("runs/*.pt")))
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--max-params", type=int, default=10 ** 6)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ho = D.make_holdout(args.n, dev, prop_p=0.0, seed=99991)
    ch = chain_cases(args.n, dev)

    rows = []
    print("%-10s %6s %9s %9s %7s %7s %7s %8s  %s"
          % ("ckpt", "params", "uniform", "chains", "ent", "on-gen", "readout", "frozen", "cfg"))
    for p in args.ckpts:
        try:
            r = audit(p, dev, ho, ch)
        except Exception as e:
            print("%-10s  skipped (%s)" % (os.path.basename(p), type(e).__name__))
            continue
        rows.append(r)
        print("%-10s %6d %8.3f%% %8.3f%% %7.3f %6.1f%% %6.1f%% %7.2f%% %s %s"
              % (r["path"], r["params"], 100 * r["acc"], 100 * r["chain"], r["ent"],
                 100 * r["argmax"], 100 * r["readout"], 100 * r["frozen"],
                 "ACCEPT" if r["ok"] else "      ", r["cfg"]))
    good = sorted([r for r in rows if r["ok"]], key=lambda r: r["params"])
    print("\n%d checkpoint(s) pass every gate:" % len(good))
    for r in good:
        print("  %-10s %4d params  uniform %.3f%%  chains %.3f%%  %s"
              % (r["path"], r["params"], 100 * r["acc"], 100 * r["chain"], r["cfg"]))


if __name__ == "__main__":
    main()
