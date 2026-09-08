"""Read the circuit out of a trained checkpoint.

The claim under test is that the attention layer implements the carry look-up:
place ``i`` should attend to the nearest place ``j < i`` that is *not*
carry-transparent (``a_j + b_j != 9``), and read off whether that place
*generates* (``a_j + b_j >= 10``).  This script measures that directly instead
of taking it on faith.

Run:  python inspect_ckpt.py runs/L9.pt
"""

import argparse

import torch

import data as D
import model_src
from model_src import AdderTransformer


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = AdderTransformer(
        ck["d_model"], ck["layers"], pos_mode=ck["pos_mode"], tie_head=ck["tie_head"],
        head_bias=ck["head_bias"], q_bias=ck["q_bias"], learn_scale=ck["learn_scale"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def to_tokens(a, b):
    pad = torch.zeros((a.shape[0], 1), dtype=torch.long, device=a.device)
    return torch.cat([pad, a.long(), pad], 1), torch.cat([pad, b.long(), pad], 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=8192)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ck = load(args.ckpt, dev)
    nparams = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"{args.ckpt}: {nparams} params  d_model={ck['d_model']}  layers={ck['layers']}")
    print(f"  recorded holdout={ck.get('holdout_acc', 0)*100:.4f}% "
          f"chain={ck.get('hard_acc', 0)*100:.4f}%")

    model_src.STORE_ATTN = True
    a, b = D.holdout_chain(args.n, dev, seed=606)   # chain cases: carry must travel
    logits = model(*to_tokens(a, b))

    s = (a + b).to(torch.long)
    s = torch.cat([torch.zeros_like(s[:, :1]), s, torch.zeros_like(s[:, :1])], 1)  # (B,10)
    transparent = s == 9
    generates = s >= 10
    # ground-truth carry into place i
    carry_in = torch.zeros_like(s, dtype=torch.bool)
    c = torch.zeros(s.shape[0], dtype=torch.bool, device=dev)
    for i in range(10):
        carry_in[:, i] = c
        c = generates[:, i] | (transparent[:, i] & c)

    for li, blk in enumerate(model.blocks):
        if blk.attn is None:
            continue
        att = blk.attn.attn                       # (B, H, T, T)
        print("-" * 70)
        print(f"attention layer {li}: heads={att.shape[1]} d_head={blk.attn.d_head}")
        print(f"  max std of a weight across inputs: {att.std(0).max():.4f}")
        print(f"  mean attention entropy (rows 1..9): "
              f"{(-(att[:, :, 1:] * att[:, :, 1:].clamp_min(1e-9).log()).sum(-1)).mean():.4f} nats")

        # Does the argmax land on the nearest non-transparent place below i?
        agree = tot = 0
        readout = 0
        for i in range(1, 10):
            j = torch.arange(i, device=dev)
            tgt = ((~transparent[:, :i]).float() * (j + 1)).argmax(-1)   # nearest, prefers big j
            pick = att[:, :, i, :i].mean(1).argmax(-1)
            agree += int((pick == tgt).sum())
            tot += a.shape[0]
            # mass the head puts on generating places should track the true carry
            mass = (att[:, :, i, :i].mean(1) * generates[:, :i].float()).sum(-1)
            readout += int(((mass > 0.5) == carry_in[:, i]).sum())
        print(f"  argmax == nearest non-transparent place below i: {100*agree/tot:.2f}%")
        print(f"  (attention mass on generating places > 0.5) == true carry_in: "
              f"{100*readout/tot:.2f}%")

    # What the embedding looks like: is it a clock code?
    print("-" * 70)
    E = model.emb.weight
    G = (E @ E.T)
    G = G / G.diag().sqrt().unsqueeze(0) / G.diag().sqrt().unsqueeze(1)
    print("  embedding cosine similarity by digit distance |i-j|:")
    for d in range(6):
        vals = [G[i, j].item() for i in range(10) for j in range(10) if abs(i - j) == d]
        print(f"    d={d}: mean cos = {sum(vals)/len(vals):+.3f}")

    # Per-digit accuracy on chain cases
    tgt = D.targets_from_values(D.digits_to_value(a), D.digits_to_value(b))
    pred = logits[:, 1:, :].argmax(-1)
    print("-" * 70)
    print("  chain-case per-output-digit error: "
          f"{['%.4f' % x for x in (pred != tgt).float().mean(0).tolist()]}")
    print(f"  chain-case exact match: {100*(pred == tgt).all(-1).float().mean():.4f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
