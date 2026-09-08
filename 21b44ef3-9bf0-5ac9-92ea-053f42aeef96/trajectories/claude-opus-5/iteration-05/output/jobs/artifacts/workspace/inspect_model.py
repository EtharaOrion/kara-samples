"""Dump what a trained member actually computes, so the next parameter cut is
an informed one rather than a guess.

  python inspect_model.py --ckpt work/p68.pt [--member N]
"""

import argparse

import torch

import data
import model_src
import verify


def load(ckpt, member=None):
    ck = torch.load(ckpt, map_location='cpu')
    acc = ck['acc']
    idx = int(acc.argmax()) if member is None else member
    m = model_src.Adder(ck['cfg'])
    with torch.no_grad():
        for k, p in m.named_parameters():
            p.copy_(ck['params'][k][idx])
    return m.double().eval(), float(acc[idx]), idx, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--member', type=int, default=None)
    args = ap.parse_args()
    m, acc, idx, ck = load(args.ckpt, args.member)
    cfg = m.cfg
    d = cfg['d']
    print(f'== member {idx}  acc {acc:.6f}  params {sum(p.numel() for p in m.parameters())}')
    print('cfg', cfg)
    for k, p in m.named_parameters():
        print(f'  {k:8s} {list(p.shape)}  {p.detach().flatten().tolist()}')

    code = m.code().detach()
    print('\ncode rows:')
    for i in range(10):
        print(f'  {i}: {[round(v,5) for v in code[i].tolist()]}')

    # x1 as a function of the digit pair
    da = torch.arange(10).repeat_interleave(10)
    db = torch.arange(10).repeat(10)
    seq_a = da[:, None].expand(-1, model_src.N_POS).contiguous()
    seq_b = db[:, None].expand(-1, model_src.N_POS).contiguous()
    x = code[seq_a] + code[seq_b]
    x1 = x + m._ffn(x, getattr(m, 'f1_w', None), m.f1_b, m.f1_o,
                    cfg['f1_id_in'], cfg['f1_out_axis'], d)
    x1 = x1[:, 0]
    s = (da + db)
    print('\n s : x1 (mean over pairs with that sum)  [spread within the group]')
    for v in range(19):
        sel = x1[s == v]
        mu = sel.mean(0)
        sp = (sel - mu).abs().max().item()
        print(f'  {v:2d}: {[round(z,4) for z in mu.tolist()]}   spread {sp:.2e}')

    print('\nattention params: '
          f"q_b={getattr(m,'q_b',None)} q_w={getattr(m,'q_w',None)} "
          f"k_w={getattr(m,'k_w',None)} "
          f"alibi={getattr(m,'alibi', getattr(m,'alibi_c',None))} "
          f"s_b={getattr(m,'s_b',None)} w_o={getattr(m,'w_o',None)}")

    # attention maps on a chain example
    ex = [(19999999, 10000001), (12345678, 87654321), (11111111, 22222222)]
    for a, b in ex:
        ai = torch.tensor([a]); bi = torch.tensor([b])
        dda, ddb, y = data.from_ints(ai, bi, 'cpu')
        lg, p = verify.forward_parts(m, dda, ddb)
        pred = lg[0, 1:].argmax(-1)
        got = sum(int(pred[i]) * 10 ** i for i in range(9))
        print(f'\n{a}+{b}={a+b} -> {got} {"OK" if got==a+b else "WRONG"}')
        for i in range(model_src.N_POS):
            row = p[0, i, :i + 1]
            print(f'   pos{i}: ' + ' '.join(f'{float(v):.3f}' for v in row))

    # f2 shape
    print('\nf2 transfer (axis 0 in, axis 0 out):')
    lo = float(x1[..., 0].min()) - 2
    hi = float(x1[..., 0].max()) + 2
    grid = torch.linspace(lo, hi, 41).double()
    xg = torch.zeros(41, d, dtype=torch.float64)
    xg[:, 0] = grid
    yg = xg + m._ffn(xg, getattr(m, 'f2_w', None), m.f2_b, m.f2_o,
                     cfg['f2_id_in'], cfg['f2_out_axis'], d)
    for i in range(0, 41, 2):
        print(f'   {float(grid[i]):9.4f} -> {[round(z,4) for z in yg[i].tolist()]}')


if __name__ == '__main__':
    main()
