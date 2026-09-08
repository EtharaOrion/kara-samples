"""Write /workspace/submission.py from a trained ensemble checkpoint.

The graded file must contain the model and its inference path only, and import
nothing but torch -- so the weights are emitted as plain Python float literals
(a few hundred numbers) rather than any encoded blob.
"""

import argparse
import inspect
import json
import re

import torch

import model_src

HEADER = '''"""Minimal transformer that adds two 8-digit numbers.

One Macaron-style block over per-place digit-pair tokens:

    x  = code[a_i] + code[b_i]        shared 10-entry code table
    x += FFN1(x)                      pre-attention, token-wise
    x += w_o * Attention(x)           1 head, learned relative-position bias
    x += FFN2(x)                      post-attention, token-wise
    logits[d] = -|x - code[d]|^2      readout tied to the same code table

Sequence layout is LSB-first, length 10: position 0 holds a (0,0) pair that
supplies the "no carry in" default, positions 1..8 hold decimal places 0..7,
and position 9 is the carry-out slot.  Position p predicts sum digit p-1, so
all nine answer digits come from a single forward pass.

The carry chain is what the attention is for: place p has to look back past a
run of transparent places (a_i + b_i == 9) to the nearest place that either
generates a carry (a_i + b_i >= 10) or absorbs it, and that lookup depends on
the digits, not on position alone.

Weights below were produced by training this architecture from scratch
(see train_ens.py / cascade.py in the workspace).
"""

import torch
import torch.nn as nn
'''

FOOT = '''

def build_model():
    """Returns (model, metadata)."""
    model = Adder(_CFG)
    sd = dict(model.named_parameters())
    for name, flat in _W.items():
        p = sd[name]
        t = torch.tensor(flat, dtype=torch.float32).reshape(p.shape)
        with torch.no_grad():
            p.copy_(t)
    model.eval()
    meta = dict(_META)
    meta['parameters'] = sum(p.numel() for p in model.parameters())
    return model, meta


def add(model, a, b):
    """Exact sum of two 8-digit integers, decoded from one forward pass."""
    da = [0] + [(int(a) // 10 ** i) % 10 for i in range(8)] + [0]
    db = [0] + [(int(b) // 10 ** i) % 10 for i in range(8)] + [0]
    dev = next(model.parameters()).device
    ta = torch.tensor([da], dtype=torch.long, device=dev)
    tb = torch.tensor([db], dtype=torch.long, device=dev)
    with torch.no_grad():
        pred = model(ta, tb).argmax(-1)[0].tolist()
    return sum(pred[p] * 10 ** (p - 1) for p in range(1, N_POS))
'''


def class_source():
    src = inspect.getsource(model_src.Adder)
    src = src.replace(
        "cfg = default_cfg(**kw) if cfg is None else default_cfg(**cfg)",
        "cfg = dict(cfg)")
    # inference path only: drop the training-time helper
    src = re.sub(r"\n    def n_params\(self\):\n(?:.*\n)*?\n?$", "\n", src)
    src = src.replace("def __init__(self, cfg=None, **kw):",
                      "def __init__(self, cfg):")
    return src


def fmt(t):
    t = t.detach().cpu().to(torch.float32)
    if t.dim() == 0:
        return repr(float(t))
    return '[' + ', '.join(fmt(x) for x in t) + ']'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--member', type=int, default=-1)
    ap.add_argument('--out', default='/workspace/submission.py')
    ap.add_argument('--note', default='')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    cfg, params, acc = ck['cfg'], ck['params'], ck['acc']
    cfg = model_src.default_cfg(**cfg)   # fill in keys added since the ckpt
    idx = int(acc.argmax()) if args.member < 0 else args.member
    model = model_src.Adder(cfg)
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(params[k][idx])
    n = sum(p.numel() for p in model.parameters())
    print(f'member {idx}  acc {float(acc[idx]):.6f}  params {n}')

    meta = {
        'name': 'tiny-8digit-adder',
        'architecture': 'single Macaron transformer block, 1 attention head',
        'residual_width': cfg['d'],
        'sequence_length': model_src.N_POS,
        'trained': 'from scratch on synthetic 8-digit addition',
        'note': args.note,
    }
    lines = [HEADER, '', f'N_POS = {model_src.N_POS}', '',
             'PIN_ROWS = ' + repr(model_src.PIN_ROWS), '', '',
             class_source(), '', f'_CFG = {cfg!r}', '',
             f'_META = {meta!r}', '', '_W = {']
    for k, p in model.named_parameters():
        lines.append(f'    {k!r}: {fmt(params[k][idx])},')
    lines.append('}')
    lines.append(FOOT)
    src = '\n'.join(lines)
    with open(args.out, 'w') as f:
        f.write(src)
    print(f'wrote {args.out}  ({len(src)} bytes)')


if __name__ == '__main__':
    main()
