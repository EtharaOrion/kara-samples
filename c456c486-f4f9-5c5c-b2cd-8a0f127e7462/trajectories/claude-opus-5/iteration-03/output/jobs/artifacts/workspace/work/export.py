"""Write a trained ensemble member out as the standalone /workspace/submission.py."""
import argparse, os, torch, tiny

TEMPLATE = '''"""Minimal transformer that performs 14-digit decimal addition in a single forward pass.

Architecture (1 layer, 1 head, d_model={d_model}, d_head=1, d_ff={d_ff}; {npar} parameters):

  Tokenisation.  The two operands are zero padded to 15 digits and read least
  significant first.  Slot i of the 16-slot sequence carries the digit pair
  (a[i-1], b[i-1]); slot 0 is a phantom (0,0) pair that acts as the sentinel.
  Each slot is embedded by two learned tables: a per-digit code U summed over the
  two digits (U[a]+U[b], channel 0) and an embedding P of the unordered digit-pair
  token (channel 1).  Remaining channels start at zero and act as scratch space.

  Attention.  A single strictly-causal head with scalar queries and keys,
  score(i,j) = (q_i + bq)(k_j + bk) + slope * (j - i).  The scores are genuinely
  content dependent: the key is driven by the pair embedding, so a slot whose
  digits sum to 9 (a carry *propagator*) is scored down and skipped over, while
  the learned recency slope makes the query select the nearest surviving slot.
  Each output position therefore reads the nearest earlier slot that resolves the
  carry, and the value channel reports whether that slot generated one.

  MLP and readout.  A residual ReLU MLP combines the local digit sum with the
  attended carry, and {headdesc}

`add` runs exactly one forward pass and decodes the 15 digit argmaxes.
"""

import torch
import torch.nn as nn

__all__ = ["TinyAdder", "build_model", "add"]

NSLOT = 16          # 1 sentinel + 15 digit positions
NDIG = 15           # digits of the sum
D_MODEL = {d_model}
D_FF = {d_ff}


def _pair_token_table():
    """Ordered digit pair -> unordered pair-token id (0..54).  Pure index arithmetic."""
    d = torch.arange(10)
    lo = torch.minimum(d[:, None], d[None, :])
    hi = torch.maximum(d[:, None], d[None, :])
    return (lo * (21 - lo)) // 2 + (hi - lo)


class TinyAdder(nn.Module):
    """Single-layer decoder-only transformer over 16 digit-pair slots."""

    def __init__(self):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(10))              # per-digit code, summed over operands
        self.P = nn.Parameter(torch.zeros(55))              # unordered digit-pair embedding
        self.wq = nn.Parameter(torch.zeros(D_MODEL))
        self.wk = nn.Parameter(torch.zeros(D_MODEL))
        self.wv = nn.Parameter(torch.zeros(D_MODEL))
        self.wo = nn.Parameter(torch.zeros(D_MODEL))
        self.bq = nn.Parameter(torch.zeros(()))
        self.bk = nn.Parameter(torch.zeros(()))
        self.slope = nn.Parameter(torch.zeros(()))          # learned ALiBi-style recency term
        self.W1 = nn.Parameter(torch.zeros(D_MODEL, D_FF))
        self.b1 = nn.Parameter(torch.zeros(D_FF))
        self.W2 = nn.Parameter(torch.zeros(D_FF, D_MODEL))
        self.b2 = nn.Parameter(torch.zeros(D_MODEL))
{head_init}
        self.register_buffer("pair_token", _pair_token_table(), persistent=False)
        idx = torch.arange(NSLOT)
        self.register_buffer("rel", (idx[None, :] - idx[:, None]).float(), persistent=False)
        # Strictly causal: the carry into slot i is determined by slots strictly
        # before i, so a slot must not attend to itself.  Slot 0 self-attends only
        # to keep its softmax row finite.
        causal = idx[None, :] < idx[:, None]
        causal[0, 0] = True
        self.register_buffer("causal", causal, persistent=False)
        self._load()

    def _load(self):
        with torch.no_grad():
            for name, values in _WEIGHTS.items():
                p = getattr(self, name)
                p.copy_(torch.tensor(values, dtype=torch.float32).reshape(p.shape))

    def forward(self, a_dig, b_dig):
        """a_dig, b_dig: int64 [B, 16] digit ids (slot 0 is the (0,0) sentinel).

        Returns logits [B, 15, 10] for the digits of the sum, least significant first.
        """
        pair = self.pair_token[a_dig, b_dig]                        # [B,16]
        c = self.U[a_dig] + self.U[b_dig]                           # compositional digit code
        e = self.P[pair]                                            # pair-token embedding
        chans = [c, e] + [torch.zeros_like(c)] * (D_MODEL - 2)
        x = torch.stack(chans, dim=-1)                              # [B,16,D_MODEL]

        q = x @ self.wq + self.bq
        k = x @ self.wk + self.bk
        scores = q[:, :, None] * k[:, None, :] + self.slope * self.rel
        scores = scores.masked_fill(~self.causal, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        o = att @ (x @ self.wv)[..., None]                          # [B,16,1]
        x = x + o * self.wo

        h = torch.relu(x @ self.W1 + self.b1)
        x = x + h @ self.W2 + self.b2

        x = x[:, 1:, :]                                             # slots 1..15 hold the answer
{head_fwd}
        return logits


def build_model():
    model = TinyAdder()
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    metadata = {{
        "name": "TinyAdder",
        "task": "addition of two integers in [0, 99999999999999]",
        "architecture": "1-layer 1-head decoder-only transformer, d_model={d_model}, d_head=1, d_ff={d_ff}",
        "n_parameters": n,
        "tokenization": "16 slots of (digit_a, digit_b) pairs, least significant first, 0-padded",
        "decoding": "single forward pass, per-position argmax over 10 digit classes",
    }}
    return model, metadata


@torch.no_grad()
def add(model, a, b):
    """Exact sum of two integers in [0, 99999999999999], from one forward pass."""
    a, b = int(a), int(b)
    da = [0] + [(a // 10 ** i) % 10 for i in range(NDIG)]
    db = [0] + [(b // 10 ** i) % 10 for i in range(NDIG)]
    dev = model.U.device
    a_dig = torch.tensor([da], dtype=torch.long, device=dev)
    b_dig = torch.tensor([db], dtype=torch.long, device=dev)
    digits = model(a_dig, b_dig).argmax(-1)[0].tolist()
    return sum(d * 10 ** i for i, d in enumerate(digits))


# --------------------------------------------------------------------------- weights
# Trained with AdamW + OneCycle on freshly sampled digit pairs; see the companion
# training script.  Values are exact float32 round-trips of the trained tensors.
_WEIGHTS = {weights}
'''

HEAD_INIT = {
    "linear_nb": '        self.Wr = nn.Parameter(torch.zeros(D_MODEL, 10))    # digit readout',
    "linear": '        self.Wr = nn.Parameter(torch.zeros(D_MODEL, 10))    # digit readout\n'
              '        self.br = nn.Parameter(torch.zeros(10))',
    "centres": '        self.wr = nn.Parameter(torch.zeros(D_MODEL))        # scalar answer channel\n'
               '        self.C = nn.Parameter(torch.zeros(10))              # digit class centres\n'
               '        self.tau = nn.Parameter(torch.zeros(()))',
    "tied": '        self.wr = nn.Parameter(torch.zeros(D_MODEL))        # scalar answer channel\n'
            '        self.tau = nn.Parameter(torch.zeros(()))',
}
HEAD_FWD = {
    "linear_nb": "        logits = x @ self.Wr",
    "linear": "        logits = x @ self.Wr + self.br",
    "centres": "        y = x @ self.wr                                             # scalar answer channel\n"
               "        logits = -(self.tau ** 2) * (y[..., None] - self.C) ** 2    # distance to class centres",
    "tied": "        y = x @ self.wr                                             # scalar answer channel\n"
            "        logits = -(self.tau ** 2) * (y[..., None] - self.U) ** 2    # distance to the digit code",
}
HEAD_DESC = {
    "linear_nb": "a bias-free linear readout scores the ten digit classes.",
    "linear": "a linear readout scores the ten digit classes.",
    "centres": "each position is scored against ten learned digit centres by squared distance.",
    "tied": "each position is scored by squared distance against the same digit code U used\n  by the input embedding, so the readout costs one extra parameter.",
}


def fmt(t):
    t = t.detach().float().cpu().reshape(-1).tolist()
    return "[" + ", ".join(repr(v) for v in t) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=-1)
    ap.add_argument("--out", default="/workspace/submission.py")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = tiny.Cfg(**{k: v for k, v in ck["cfg"].items()})
    i = ck["best_idx"] if args.idx < 0 else args.idx
    P = {k: v[i] for k, v in ck["params"].items()}

    order = ["U", "P", "wq", "wk", "wv", "wo", "bq", "bk", "slope", "W1", "b1", "W2", "b2"]
    order += {"linear": ["Wr", "br"], "linear_nb": ["Wr"], "centres": ["wr", "C", "tau"], "tied": ["wr", "tau"]}[cfg.head]
    body = ",\n    ".join(f'"{k}": {fmt(P[k])}' for k in order if k in P)
    src = TEMPLATE.format(
        d_model=cfg.d_model, d_ff=cfg.d_ff, npar=tiny.param_count(cfg),
        head_init=HEAD_INIT[cfg.head], head_fwd=HEAD_FWD[cfg.head], headdesc=HEAD_DESC[cfg.head],
        weights="{\n    " + body + ",\n}\n")
    if "b2" not in P:
        src = src.replace("        self.b2 = nn.Parameter(torch.zeros(D_MODEL))\n", "")
        src = src.replace("x = x + h @ self.W2 + self.b2", "x = x + h @ self.W2")
    if "bk" not in P:
        src = src.replace("        self.bk = nn.Parameter(torch.zeros(()))\n", "")
        src = src.replace("k = x @ self.wk + self.bk", "k = x @ self.wk")
        src = src.replace("score(i,j) = (q_i + bq)(k_j + bk)", "score(i,j) = (q_i + bq) * k_j")
    with open(args.out, "w") as f:
        f.write(src)
    print(f"wrote {args.out} from {args.ckpt} member {i} "
          f"(unif {ck['acc'][i]:.4f} stress {ck['sacc'][i]:.4f} step {ck['step']})")


if __name__ == "__main__":
    main()
