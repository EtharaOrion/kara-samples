"""Admissible-band study for the shipped model's non-learned constants.

The shipped file keeps four numbers as buffers -- the sharp bank slope, the
gentle bank slope, the key contrast and the recency bias -- on the claim that
each is a shape constant of the circuit rather than a value fitted to the
arithmetic.  This script tests that claim the only way that means anything:
hold the eleven *learned* weights fixed at their shipped values and sweep each
constant, reporting the range over which the whole-domain certificate still
holds and all 3^8 carry-class patterns still come out exact.

A constant that only works at one value would be doing arithmetic.  A constant
with a wide band is doing what a slope or a temperature does.
"""
import argparse, importlib.util, json
import torch
import certify


def load(path):
    spec = importlib.util.spec_from_file_location('cand', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, _ = mod.build_model()
    return mod, model


def sweep(model, name, values, setter):
    """-> list of (value, certified, exact_patterns) with the weights untouched."""
    code = model.codes().detach().double()
    knee = model.knee.detach().double()
    fold = float(model.fold.detach())
    out = []
    for v in values:
        cfg = dict(bank_w=tuple(float(x) for x in model.bank_w),
                   key_w=tuple(float(x) for x in model.key_w),
                   val_w=tuple(float(x) for x in model.val_w),
                   lam=float(model.lam))
        setter(cfg, v)
        ok, rep = certify.certify(code, knee, fold, **cfg, verbose=False)
        saved = {k: getattr(model, k).clone() for k in ('bank_w', 'key_w', 'val_w', 'lam')}
        with torch.no_grad():
            model.bank_w.copy_(torch.tensor(cfg['bank_w']))
            model.key_w.copy_(torch.tensor(cfg['key_w']))
            model.lam.copy_(torch.tensor(cfg['lam']))
            got, tot = certify.exhaustive_patterns(model, 8)
        with torch.no_grad():
            for k, t in saved.items():
                getattr(model, k).copy_(t)
        out.append((float(v), bool(ok), got, tot, rep.get('notch_depth', 0.0)))
    return out


def band(rows, shipped):
    """Widest run of certified+exact values containing the shipped one."""
    good = [r[0] for r in rows if r[1] and r[2] == r[3]]
    if not good:
        return None
    lo = hi = shipped
    vals = [r[0] for r in rows]
    i = vals.index(min(vals, key=lambda v: abs(v - shipped)))
    j = i
    while j >= 0 and rows[j][1] and rows[j][2] == rows[j][3]:
        lo = rows[j][0]; j -= 1
    j = i
    while j < len(rows) and rows[j][1] and rows[j][2] == rows[j][3]:
        hi = rows[j][0]; j += 1
    return lo, hi


def grid(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='/workspace/submission.py')
    ap.add_argument('--out', default='runs/band_study.json')
    a_ = ap.parse_args()
    mod, model = load(a_.path)
    print('learned weights held fixed:',
          {n: [round(float(x), 5) for x in p.flatten()] for n, p in model.named_parameters()})
    sh = dict(sharp=float(model.bank_w[0]), gentle=float(model.bank_w[1]),
              contrast=float(model.key_w[1]), recency=float(model.lam))
    print('shipped constants:', sh, '\n')

    tests = [
        ('gentle bank slope bank_w[1]', 'gentle', grid(0.4, 2.6, 45),
         lambda c, v: c.__setitem__('bank_w', (c['bank_w'][0], v))),
        ('sharp bank slope bank_w[0]', 'sharp', grid(1.5, 80.0, 45),
         lambda c, v: c.__setitem__('bank_w', (v, c['bank_w'][1]))),
        ('key contrast key_w', 'contrast', grid(20.0, 4000.0, 45),
         lambda c, v: c.__setitem__('key_w', (-v, v))),
        ('recency bias lam', 'recency', grid(-40.0, -0.5, 45),
         lambda c, v: c.__setitem__('lam', v)),
    ]
    report = {}
    for label, key, vals, setter in tests:
        rows = sweep(model, key, vals, setter)
        b = band(rows, sh[key])
        ratio = (b[1] / b[0]) if b and b[0] > 0 and b[1] > 0 else None
        if b and b[0] < 0:
            ratio = b[0] / b[1]
        report[key] = dict(shipped=sh[key], band=b, ratio=ratio,
                           rows=[dict(v=r[0], cert=r[1], exact=f'{r[2]}/{r[3]}',
                                      notch=round(r[4], 2)) for r in rows])
        print(f'{label}: shipped {sh[key]:g}, certified+exact over '
              f'[{b[0]:g}, {b[1]:g}]' + (f'  ({ratio:.1f}x range)' if ratio else ''))
        ok = [r[0] for r in rows if r[1] and r[2] == r[3]]
        print(f'   swept {len(rows)} values in [{vals[0]:g}, {vals[-1]:g}], '
              f'{len(ok)} of them fully exact on all 3^8 patterns')
    json.dump(report, open(a_.out, 'w'), indent=1)
    print('\nwrote', a_.out)


if __name__ == '__main__':
    main()
