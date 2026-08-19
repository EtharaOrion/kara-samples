"""Exact parameter folds applied to a converged checkpoint (each is a reparametrisation
that leaves the model's function unchanged), plus checkpoint surgery helpers."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))


def load(path):
    return torch.load(path, map_location='cpu')


def pick(ck, cells):
    """Keep a subset of ensemble cells."""
    out = dict(ck)
    out['params'] = {k: v[cells].clone() for k, v in ck['params'].items()}
    return out


def fold_unit_scale(ck):
    """relu(w*z+b)*v  ==  |w| * relu(sign(w)*z + b/|w|) * v.
    Push |w| into the two output weights so W1a keeps only its signs, which then
    become fixed architecture constants instead of parameters."""
    P = ck['params']
    w = P['W1a']
    aw = w.abs().clamp(min=1e-12)
    P['b1a'] = P['b1a'] / aw
    P['W2a0'] = P['W2a0'] * aw
    if 'W2a1' in P:
        P['W2a1'] = P['W2a1'] * aw
    else:
        P['W2a1p'] = P['W2a1p'] * aw[:, 1:]
    sign = torch.sign(w)
    P['W1a'] = sign
    ck['fold_sign_per_cell'] = sign
    return ck


def tie_analytic(ck):
    """Fold the separate readout centres V back onto the embedding table (exact).

    Untied optimum:  U[d]=a*d+b, V[d]=a'*d+b', y = z_U + w1*c_in + w2*c_out ~ V[digit].
    Matching coefficients gives a == a' and the constant relation 2b + (w1+w2)*v0 = b'.
    A tied model needs code W with intercept B = -(w1+w2)*v0 = 2b - b', i.e. W = V + q
    where q is the intercept of the fit z_U = p*z_V + q.  Rescaling the feature layer by
    (p, q) keeps every ReLU activation bit-identical, so attention and c2 are untouched and
    y shifts by exactly q, which is the same shift the centres received.
    """
    P = ck['params']
    U, V = P['U'], P['V']
    pr = [(a, b) for a in range(10) for b in range(10)]
    zu = torch.stack([U[:, a] + U[:, b] for a, b in pr], 1)
    zv = torch.stack([V[:, a] + V[:, b] for a, b in pr], 1)
    zum, zvm = zu.mean(1, keepdim=True), zv.mean(1, keepdim=True)
    p = (((zv - zvm) * (zu - zum)).sum(1, keepdim=True) / ((zv - zvm) ** 2).sum(1, keepdim=True))
    q = zum - p * zvm
    resid = (zu - (p * zv + q)).abs().max(1).values
    P['b1a'] = P['b1a'] + P['W1a'] * q * (1 - 2 * p)
    P['W1a'] = P['W1a'] * p
    P['U'] = V + q
    P.pop('V')
    print('tie: p', [round(float(x), 4) for x in p.flatten()[:4]],
          'q', [round(float(x), 3) for x in q.flatten()[:4]],
          'max affine residual', [round(float(x), 4) for x in resid[:4]])
    return ck


def tie(ck):
    """Drop the separate output centres V (they have been annealed onto U)."""
    ck['params'].pop('V', None)
    return ck


if __name__ == '__main__':
    cmd = sys.argv[1]
    ck = load(sys.argv[2])
    if cmd == 'report':
        for k, v in ck['params'].items():
            print(k, tuple(v.shape))
        print('meta', ck.get('meta'), 'nparams', ck.get('nparams'))
        if 'V' in ck['params']:
            d = (ck['params']['U'] - ck['params']['V']).abs().max(-1).values
            print('max|U-V| per cell:', [round(float(x), 4) for x in d[:8]])
    else:
        cells = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else list(range(ck['params']['U'].shape[0]))
        ck = pick(ck, cells)
        if cmd in ('tie', 'tiefold'):
            ck = tie_analytic(ck)
        if cmd in ('fold', 'tiefold'):
            ck = fold_unit_scale(ck)
        torch.save(ck, sys.argv[3])
        print('wrote', sys.argv[3], {k: tuple(v.shape) for k, v in ck['params'].items()})

def drop_value_unit(ck, k=0):
    """Remove unit k from the value projection W2a1 (exact on the domain that is read).

    After fold_unit_scale every unit has slope +-1, so on any interval where the same
    units are active c2 is affine in z.  It must be flat there, which the trained model
    achieves with sum(W2a1) == 0; the plateau is then P = sum(W2a1 * b1a) and c2 == 0
    where no unit fires.  Two entries already have enough freedom to hit (slope 0,
    plateau P) exactly, so the third is redundant: c2 is unchanged for every digit sum
    except s == 9, and s == 9 is a propagate slot, which the attention never selects as a
    key -- its value is never read.  Verified empirically by evaluating both models.
    """
    P = ck['params']
    w, b = P['W2a1'], P['b1a']
    j, l = [i for i in range(w.shape[1]) if i != k]
    plateau = (w * b).sum(1, keepdim=True)
    slope = w.sum(1, keepdim=True)
    wj = plateau / (b[:, j:j + 1] - b[:, l:l + 1])
    new = torch.zeros_like(w)
    new[:, j] = wj[:, 0]
    new[:, l] = -wj[:, 0]
    print('drop_value_unit: residual slope of old c2', [round(float(x), 5) for x in slope[:4, 0]],
          'plateau', [round(float(x), 3) for x in plateau[:4, 0]])
    if k != 0:
        raise SystemExit('only W2a1_0 is supported by the drop plumbing')
    P['W2a1p'] = new[:, 1:].contiguous()
    P.pop('W2a1')
    ck['drop'] = ('W2a1_0',)
    return ck

