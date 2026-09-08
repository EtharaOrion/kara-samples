"""Audit /workspace/submission.py: accuracy, edge cases, and whether the
attention is actually doing work.

  python verify.py [--n 1000000] [--full]
"""

import argparse
import importlib.util
import sys
import time

import torch

import data


def load_submission(path='/workspace/submission.py'):
    spec = importlib.util.spec_from_file_location('submission_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['submission_under_test'] = mod
    spec.loader.exec_module(mod)
    return mod


# --- a local copy of the forward pass, so the attention map can be replaced ---
def forward_parts(model, da, db, p_override=None):
    cfg = model.cfg
    d = cfg['d']
    code = model.code()
    x = code[da] + code[db]
    x = x + model._ffn(model._norm(x, '1'), getattr(model, 'f1_w', None), model.f1_b,
                       model.f1_o, cfg['f1_id_in'], cfg['f1_out_axis'], d,
                       getattr(model, 'f1_map', None))
    xn = model._norm(x, 'a')
    ka = cfg.get('k_axis', 0)
    k = xn @ model.k_w if cfg['k_weight'] else xn[..., ka:ka + 1]
    if cfg['q_weight']:
        q = xn @ model.q_w
        if cfg['q_bias']:
            q = q + model.q_b
    else:
        q = model.q_b.expand_as(k)
    alibi = model.alibi if cfg['alibi_fix'] == 0.0 else model.alibi_c
    att = q * k.transpose(-1, -2) + alibi * model.dist
    if cfg['self_bias']:
        att = att + model.s_b * model.eye
    att = att.masked_fill(~model.mask, -1e9)
    p = torch.softmax(att, -1)
    use = p if p_override is None else p_override.expand_as(p)
    a_out = (use @ (xn @ model.v_w)) @ model.o_w if cfg.get('attn_v') else use @ xn
    if cfg['w_o']:
        a_out = a_out * model.w_o
    x2 = x + a_out
    x2 = x2 + model._ffn(model._norm(x2, '2'), getattr(model, 'f2_w', None),
                         model.f2_b, model.f2_o, cfg['f2_id_in'],
                         cfg['f2_out_axis'], d, getattr(model, 'f2_map', None))
    logits = -((x2[..., None, :] - code) ** 2).sum(-1)
    if cfg.get('logit_scale'):
        logits = logits * model.tau
    return logits, p


def batched_acc(model, da, db, y, p_override=None, chunk=65536):
    ok = 0
    n = da.shape[0]
    for i in range(0, n, chunk):
        lg, _ = forward_parts(model, da[i:i + chunk], db[i:i + chunk], p_override)
        ok += int((lg[:, 1:].argmax(-1) == y[i:i + chunk, 1:]).all(-1).sum())
    return ok / n


def struct_cases(device, k=4, seed=0):
    """One batch covering all 3^8 generate / transparent / absorb patterns."""
    g = torch.Generator(device=device).manual_seed(seed)
    P = 8
    pat = torch.arange(3 ** P, device=device)
    cls = torch.stack([(pat // 3 ** i) % 3 for i in range(P)], 1)      # (N, 8)
    cls = cls.repeat(k, 1)
    N = cls.shape[0]
    a = torch.randint(0, 10, (N, P), device=device, generator=g)
    b = torch.randint(0, 10, (N, P), device=device, generator=g)
    # class 0: absorb (sum <= 8) | 1: transparent (sum == 9) | 2: generate (>= 10)
    absorb = a.clamp(max=8)
    b_abs = (torch.rand((N, P), device=device, generator=g) * (9 - absorb)).long()
    b_gen = 10 - a.clamp(min=1) + (torch.rand((N, P), device=device, generator=g)
                                   * (a.clamp(min=1))).long()
    a = torch.where(cls == 2, a.clamp(min=1), a)
    a = torch.where(cls == 0, absorb, a)
    b = torch.where(cls == 0, b_abs, b)
    b = torch.where(cls == 1, 9 - a, b)
    b = torch.where(cls == 2, b_gen.clamp(0, 9), b)
    top = torch.randint(1, 10, (N, 2), device=device, generator=g)
    a[:, P - 1] = torch.where(a[:, P - 1] == 0, top[:, 0], a[:, P - 1])
    b[:, P - 1] = torch.where(b[:, P - 1] == 0, top[:, 1], b[:, P - 1])
    ai, bi = data.to_int(a, device), data.to_int(b, device)
    return data.pack(a, b, ai, bi, device)


EDGE = [(10000000, 10000000), (99999999, 99999999), (10000000, 99999999),
        (19999999, 10000001), (12345678, 87654321), (99999999, 10000001),
        (55555555, 44444445), (10000001, 89999999), (98765432, 12345678),
        (11111111, 88888889), (50000000, 50000000), (99999998, 10000001)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--path', default='/workspace/submission.py')
    ap.add_argument('--full', action='store_true')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    sub = load_submission(args.path)
    model, meta = sub.build_model()
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    print(f'params {n_par}   float buffers {n_buf}   meta {meta}')

    # 1. the interface itself
    t0 = time.time()
    bad = [(a, b) for a, b in EDGE if sub.add(model, a, b) != a + b]
    print(f'edge cases: {len(EDGE) - len(bad)}/{len(EDGE)} ok' +
          (f'  FAILED {bad}' if bad else ''))

    g = torch.Generator(device=dev).manual_seed(777)
    ai = torch.randint(10_000_000, 100_000_000, (256,), device=dev, generator=g)
    bi = torch.randint(10_000_000, 100_000_000, (256,), device=dev, generator=g)
    mism = sum(1 for a, b in zip(ai.tolist(), bi.tolist())
               if sub.add(model, a, b) != a + b)
    print(f'add() on 256 random pairs: {256 - mism}/256 exact  ({time.time()-t0:.1f}s)')

    model = model.to(dev)
    # 2. bulk accuracy on unseen (held-out hash bucket) uniform pairs
    tot, ok = 0, 0.0
    for _ in range(max(1, args.n // 100000)):
        da, db, y = data.sample_uniform_pairs(min(args.n, 100000), dev, g)
        ok += batched_acc(model, da, db, y) * da.shape[0]
        tot += da.shape[0]
    print(f'held-out uniform pairs: {ok/tot:.6f}  (n={tot})')

    # 3. structured carry patterns: all 3^8 generate/transparent/absorb masks
    sa, sb, sy = struct_cases(dev, k=4)
    acc_s = batched_acc(model, sa, sb, sy)
    print(f'all 6561 carry patterns x4: {acc_s:.6f}  (n={sa.shape[0]})')

    # 4. hardest chains
    ca, cb, cy = data.sample(200000, dev, g, (0.0, 0.3, 0.7))
    print(f'carry-chain enriched:  {batched_acc(model, ca, cb, cy):.6f}')

    # 5. is the attention input-dependent and load-bearing?
    _, p = forward_parts(model, sa[:4096], sb[:4096])
    argm = p.argmax(-1)
    varies = (argm != argm[:1]).any(0).float().mean()
    print(f'attention argmax varies across inputs at {varies*100:.1f}% of (i,j) slots;'
          f' map std {p.std(0).max():.4f}')
    pbar = p.mean(0, keepdim=True)
    print(f'FROZEN-attention accuracy (batch-mean map): '
          f'{batched_acc(model, sa, sb, sy, p_override=pbar):.6f}   '
          f'uniform: {batched_acc(model, da, db, y, p_override=pbar):.6f}')

    # 6. numerics: float64 and cpu agreement
    m64 = sub.build_model()[0].to(dev).double()
    l32, _ = forward_parts(model, sa[:65536], sb[:65536])
    l64, _ = forward_parts(m64, sa[:65536], sb[:65536])
    agree = (l32[:, 1:].argmax(-1) == l64[:, 1:].argmax(-1)).all().item()
    mcpu = sub.build_model()[0]
    lcpu, _ = forward_parts(mcpu, sa[:20000].cpu(), sb[:20000].cpu())
    agree_cpu = (lcpu[:, 1:].argmax(-1) == l32[:20000, 1:].argmax(-1).cpu()).all().item()
    print(f'float32 == float64 argmax: {agree}   cpu == cuda: {agree_cpu}')

    # 7. decision margin: distance to the nearest wrong prototype, in prototype steps
    lg, _ = forward_parts(m64, sa, sb)
    lg = lg[:, 1:]
    tgt = lg.gather(-1, sy[:, 1:, None]).squeeze(-1)
    other = lg.masked_fill(
        torch.nn.functional.one_hot(sy[:, 1:], 10).bool(), -1e30).max(-1).values
    print(f'min logit margin over structured set: {float((tgt - other).min()):.4g}')

    if args.full:
        n = 0
        okc = 0.0
        for i in range(20):
            da, db, y = data.sample_uniform_pairs(200000, dev, g)
            okc += batched_acc(model, da, db, y) * da.shape[0]
            n += da.shape[0]
        print(f'FULL held-out uniform: {okc/n:.7f} (n={n})')


if __name__ == '__main__':
    main()
