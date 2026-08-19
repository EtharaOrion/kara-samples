"""Collect the best cells across runs into one checkpoint (optionally replicated)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from core import Ens, sample_batch, stress_batch, targets, to_onehot

dev = 'cuda'


def score_ckpt(path):
    ck = torch.load(path, map_location='cpu')
    P = ck['params']
    M = P['U'].shape[0]
    W = P['b1a'].shape[1]
    m = Ens(M, width=W, untied=('V' in P), fold_sign=ck.get('fold_sign'), drop=ck.get('drop') or ())
    with torch.no_grad():
        for k, v in m.named_parameters():
            v.copy_(P[k])
    m = m.to(dev)
    g = torch.Generator(device=dev).manual_seed(31337)
    tot = torch.zeros(M, device=dev)
    for fn in (sample_batch, stress_batch):
        a, b = fn(8192, dev, g)
        t = targets(a, b, dev); oh = to_onehot(a, b, dev)
        with torch.no_grad():
            tot += (m(oh).argmax(-1) == t.unsqueeze(0)).all(-1).float().mean(-1)
    return ck, tot / 2


if __name__ == '__main__':
    out = sys.argv[1]
    topk = int(sys.argv[2])
    rep = int(sys.argv[3])
    paths = sys.argv[4:]
    pool = []
    for p in paths:
        ck, sc = score_ckpt(p)
        order = sc.argsort(descending=True)[:topk]
        print(os.path.basename(p), 'top:', [(int(i), round(float(sc[i]), 4)) for i in order])
        for i in order:
            pool.append((float(sc[i]), {k: v[int(i)].clone() for k, v in ck['params'].items()},
                         ck.get('fold_sign'), ck.get('drop')))
    pool.sort(key=lambda x: -x[0])
    pool = pool[:topk]
    print('kept', [round(p[0], 4) for p in pool])
    keys = pool[0][1].keys()
    params = {k: torch.stack([p[1][k] for p in pool for _ in range(rep)]) for k in keys}
    torch.save({'params': params, 'fold_sign': pool[0][2], 'drop': pool[0][3],
                'scores': [p[0] for p in pool]}, out)
    print('wrote', out, {k: tuple(v.shape) for k, v in params.items()})
