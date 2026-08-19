"""Write /workspace/submission.py from a saved training run."""

import argparse
import types

import torch

from model_def import TinyAdder
from train import emit, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--eval_n', type=int, default=2_000_000)
    a = ap.parse_args()

    ck = torch.load(f'runs/{a.tag}.pt', map_location='cpu', weights_only=False)
    c = ck['cfg']
    for k, v in (('sym', False), ('qkb', False), ('addemb', False), ('h0', 6), ('hyb', False), ('nobout', False), ('npair', None), ('feat', 0)):
        c.setdefault(k, v)
    cfg = types.SimpleNamespace(**c)
    model = TinyAdder(c['d'], c['dh'], c['ff'], sym=c.get('sym', False),
                      qkb=c.get('qkb', False), addemb=c.get('addemb', False),
                      h0=c.get('h0', 6), hyb=c.get('hyb', False),
                      bout=not c.get('nobout', False),
                      npair=c.get('npair'), feat=c.get('feat', 0))
    model.load_state_dict(ck['sd'])
    model.eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev)

    n = sum(p.numel() for p in model.parameters())
    acc_u = evaluate(model, dev, n=a.eval_n, seed=20260813, hard=False)
    acc_h = evaluate(model, dev, n=a.eval_n, seed=20260814, hard=True)
    print(f'{a.tag}: {n} params  uniform={acc_u:.6f}  stress={acc_h:.6f}')
    emit(model.cpu(), cfg, n, acc_u, acc_h)


if __name__ == '__main__':
    main()
