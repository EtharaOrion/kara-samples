"""Diagnostics: accuracy breakdown + evidence that attention does real work."""

import sys
import torch

from model_def import TinyAdder
from data import gen_batch, gen_uniform, sum_digits


def load(tag):
    ck = torch.load(f'runs/{tag}.pt', map_location='cpu', weights_only=False)
    c = ck['cfg']
    m = TinyAdder(c['d'], c['dh'], c['ff'], sym=c.get('sym', False),
                  qkb=c.get('qkb', False), addemb=c.get('addemb', False),
                  h0=c.get('h0', 6), hyb=c.get('hyb', False),
                  bout=not c.get('nobout', False), npair=c.get('npair'),
                  feat=c.get('feat', 0))
    m.load_state_dict(ck['sd'])
    m.eval()
    return m, ck


def carry_chain_len(a, b):
    """Longest run of consecutive digit positions with a_i+b_i == 9."""
    prop = ((a + b) == 9).int()
    best = torch.zeros(a.shape[0], dtype=torch.int32)
    cur = torch.zeros_like(best)
    for i in range(a.shape[1]):
        cur = (cur + 1) * prop[:, i]
        best = torch.maximum(best, cur)
    return best


def report(tag):
    m, ck = load(tag)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = m.to(dev)
    n = sum(p.numel() for p in m.parameters())
    print(f'=== {tag}: {n} params  cfg={ck["cfg"]["d"]}/{ck["cfg"]["dh"]}/'
          f'{ck["cfg"]["ff"]} sym={ck["cfg"].get("sym")} ===')

    g = torch.Generator(device=dev).manual_seed(7)
    for name, fn in (('uniform', gen_uniform), ('stress', gen_batch)):
        ok = tot = 0
        perpos = torch.zeros(15, device=dev)
        chain_ok = {}
        for _ in range(20):
            a, b, y = fn(50_000, dev, g)
            with torch.no_grad():
                p = m(a, b).argmax(-1)
            good = (p == y).all(-1)
            ok += good.sum().item()
            tot += a.shape[0]
            perpos += (p != y).float().sum(0)
            cl = carry_chain_len(a.cpu(), b.cpu())
            for c in range(0, 8):
                sel = (cl == c) if c < 7 else (cl >= 7)
                if sel.any():
                    d = chain_ok.setdefault(c, [0, 0])
                    d[0] += good.cpu()[sel].sum().item()
                    d[1] += int(sel.sum())
        print(f'  {name}: exact {ok/tot:.6f}  ({tot} samples)')
        print(f'    digit errors by position (LSB first): '
              f'{[int(v) for v in perpos.tolist()]}')
        print('    exact by longest propagate-chain: ' + '  '.join(
            f'{c}{"+" if c==7 else ""}:{v[0]/v[1]:.4f}' for c, v in sorted(chain_ok.items())))

    # --- attention is input-dependent -------------------------------------
    a, b, _ = gen_batch(512, dev, g)
    with torch.no_grad():
        if m.hyb:
            code = (m.digit_emb[a] + m.digit_emb[b]).unsqueeze(-1)
            if m.feat:
                pair = (torch.relu(code * m.fw + m.fb) * m.fo).sum(-1, True) + m.fbo
            else:
                pair = m.pair_emb[m.pair_idx[a, b]]
            parts = [code, pair]
            spare = m.d_model - 1 - m.npair
            if spare:
                parts.append(code.new_zeros(code.shape[:-1] + (spare,)))
            tok = torch.cat(parts, -1)
        else:
            tok = m.pair_emb[m.pair_idx[a, b]] if m.sym else m.pair_emb[a, b]
        x = torch.cat([m.bos.expand(512, 1, -1), tok], 1)
        q, k = x @ m.wq, x @ m.wk
        if m.qkb:
            q, k = q + m.bq, k + m.bk
        sc = (q @ k.transpose(1, 2)) * (m.d_head ** -0.5)
        idx = torch.arange(x.shape[1], device=dev)
        sc = sc + m.slope * (idx[None, :] - idx[:, None]).float()
        allowed = (idx[None, :] < idx[:, None]).clone()
        allowed[0, 0] = True
        att = sc.masked_fill(~allowed, float('-inf')).softmax(-1)
    tgt = att[:, -1, :]                       # attention of the top digit position
    argmax_j = tgt.argmax(-1)
    print(f'  attention @ last position: argmax source varies over '
          f'{len(argmax_j.unique())} distinct positions across 512 inputs; '
          f'per-input std of the attended index = {argmax_j.float().std():.2f}')
    print(f'  mean max attention weight {tgt.max(-1).values.mean():.4f}, '
          f'mean entropy {-(att[:,1:]*att[:,1:].clamp_min(1e-9).log()).sum(-1).mean():.4f}')
    print(f'  learned recency slope = {m.slope.item():.3f}')


if __name__ == '__main__':
    for t in sys.argv[1:]:
        report(t)
