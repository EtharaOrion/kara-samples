import argparse
import importlib.util
import random
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path('/workspace')
SPEC = importlib.util.spec_from_file_location('submission', ROOT / 'submission.py')
SUB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUB)


def digits8(x):
    places = torch.tensor([1, 10, 100, 1000, 10000, 100000, 1000000, 10000000], device=x.device)
    return (x[:, None] // places % 10).long()


def make_batch(n, device, structured=0.45):
    a = torch.randint(10_000_000, 100_000_000, (n,), device=device)
    b = torch.randint(10_000_000, 100_000_000, (n,), device=device)
    use = torch.rand(n, device=device) < structured
    idx = use.nonzero().flatten()
    if idx.numel():
        kinds = torch.randint(0, 6, (idx.numel(),), device=device)
        # Complements create carries through all eight columns.
        q = idx[kinds == 0]
        if q.numel():
            a[q] = torch.randint(10_000_000, 90_000_001, (q.numel(),), device=device)
            b[q] = 100_000_000 - a[q]
        # Both operands near the upper endpoint.
        q = idx[kinds == 1]
        if q.numel():
            a[q] = 100_000_000 - torch.randint(1, 1_000_001, (q.numel(),), device=device)
            b[q] = 100_000_000 - torch.randint(1, 1_000_001, (q.numel(),), device=device)
        # Rounded operands at randomly selected decimal scales.
        q = idx[kinds == 2]
        if q.numel():
            scales = torch.tensor([10, 100, 1000, 10000, 100000, 1000000, 10000000], device=device)
            s = scales[torch.randint(0, len(scales), (q.numel(),), device=device)]
            a[q] = (a[q] // s) * s
            b[q] = (b[q] // s) * s
            a[q].clamp_(min=10_000_000)
            b[q].clamp_(min=10_000_000)
        # Repeated decimal digits.
        q = idx[kinds == 3]
        if q.numel():
            vals = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=device)
            a[q] = vals[torch.randint(0, 9, (q.numel(),), device=device)]
            b[q] = vals[torch.randint(0, 9, (q.numel(),), device=device)]
        # Long runs of trailing nines plus a value that propagates carry.
        q = idx[kinds == 4]
        if q.numel():
            scales = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=device)
            s = scales[torch.randint(0, 7, (q.numel(),), device=device)]
            a[q] = (a[q] // s) * s + s - 1
            a[q].clamp_(max=99_999_999)
            b[q] = torch.randint(10_000_000, 100_000_000, (q.numel(),), device=device)
        # Sparse and simple patterns, including exact powers/near powers.
        q = idx[kinds == 5]
        if q.numel():
            vals = torch.tensor([10_000_000,10_000_001,10_000_010,10_000_100,10_001_000,10_010_000,10_100_000,11_000_000,20_000_000,50_000_000,90_000_000,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_999], device=device)
            a[q] = vals[torch.randint(0, len(vals), (q.numel(),), device=device)]
            b[q] = vals[torch.randint(0, len(vals), (q.numel(),), device=device)]
    ad, bd = digits8(a), digits8(b)
    operands = torch.stack((ad, bd), dim=2).reshape(n, 16)
    target = digits8(a + b)
    ninth = ((a + b) // 100_000_000).long()[:, None]
    target = torch.cat((target, ninth), dim=1)
    sentinel = torch.full((n, 1), 10, dtype=torch.long, device=device)
    inputs = torch.cat((operands, sentinel, target[:, :-1]), dim=1)
    return inputs, target, a, b


@torch.no_grad()
def evaluate(model, n, structured, batch=10000):
    model.eval()
    good = total = 0
    for _ in range((n + batch - 1) // batch):
        size = min(batch, n - total)
        inp, target, _, _ = make_batch(size, next(model.parameters()).device, structured)
        seq = inp[:, :17]
        out = []
        for _ in range(9):
            pred = model(seq)[:, -1].argmax(1)
            out.append(pred)
            seq = torch.cat((seq, pred[:, None]), 1)
        pred = torch.stack(out, 1)
        good += (pred == target).all(1).sum().item()
        total += size
    model.train()
    return good / total


def tensor_literal(t):
    # repr of nested Python lists is explicitly accepted source-level storage.
    return 'torch.tensor(' + repr(t.detach().cpu().tolist()) + ', dtype=torch.float32)'


def export(model):
    path = ROOT / 'submission.py'
    text = path.read_text()
    start = text.index('_STATE = ')
    end = text.index('\n\n\ndef build_model', start)
    entries = []
    for name, value in model.state_dict().items():
        entries.append(repr(name) + ': ' + tensor_literal(value))
    state = '_STATE = {\n    ' + ',\n    '.join(entries) + '\n}'
    path.write_text(text[:start] + state + text[end:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    model = SUB.AdderTransformer().to(device).train()
    if args.resume:
        model.load_state_dict(torch.load(ROOT / 'checkpoint.pt', weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.steps)
    for step in range(1, args.steps + 1):
        structured = 0.55
        inp, target, _, _ = make_batch(args.batch, device, structured)
        logits = model(inp)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 2000 == 0 or step == 1:
            torch.save(model.state_dict(), ROOT / 'checkpoint.pt')
            acc = evaluate(model, 20000, 0.0)
            edge = evaluate(model, 20000, 1.0)
            print(f'{step:6d} loss={loss.item():.6f} random={acc:.6%} structured={edge:.6%} lr={scheduler.get_last_lr()[0]:.2g}', flush=True)
    r = evaluate(model, 200000, 0.0)
    s = evaluate(model, 200000, 1.0)
    print(f'FINAL random={r:.6%} structured={s:.6%}', flush=True)
    export(model)
    print('exported', sum(p.numel() for p in model.parameters()), 'parameters')


if __name__ == '__main__':
    main()
