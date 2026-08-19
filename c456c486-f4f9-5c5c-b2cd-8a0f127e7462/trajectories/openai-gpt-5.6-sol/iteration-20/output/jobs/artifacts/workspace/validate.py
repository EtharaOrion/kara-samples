import copy
import importlib.util
import random
import sys

import torch

sys.path.insert(0, '/workspace')
import train


def exact_errors(model, batches, size, structured):
    errors = 0
    examples = []
    with torch.no_grad():
        for _ in range(batches):
            a, b, y = train.make_batch(size, torch.device('cuda'), structured)
            p = model(a, b).argmax(-1)
            bad = (p != y).any(1)
            errors += bad.sum().item()
            if bad.any() and len(examples) < 5:
                ids = bad.nonzero()[:5].flatten()
                examples.extend(zip(a[ids].cpu().tolist(), b[ids].cpu().tolist(), y[ids].cpu().tolist(), p[ids].cpu().tolist()))
    return errors, batches * size, examples


def from_digits(ds):
    return sum(d * 10**i for i, d in enumerate(ds))


def edge_pairs():
    limit = train.LIMIT
    pairs = [(0, 0), (limit-1, 0), (limit-1, 1), (limit-1, limit-1), (1, limit-1)]
    for start in range(14):
        p = 10**start
        for length in range(1, 15-start):
            chain = (10**length - 1) * p
            for low in (0, p-1 if p > 1 else 0):
                for prefix in (0, 3 * 10**(start+length) if start+length < 14 else 0):
                    x = min(limit-1, prefix + chain + low)
                    pairs.extend([(x, p), (p, x)])
    reps = 11_111_111_111_111
    for da in range(10):
        for db in range(10):
            pairs.extend([(da*reps, db*reps), (limit-1-da*reps, db*reps)])
    for k in range(1, 15):
        p = 10**k
        for delta in range(-9, 10):
            x = max(0, min(limit-1, p-1+delta))
            pairs.extend([(x, 1), (1, x), (x, limit-1-x)])
    return list(dict.fromkeys(pairs))


torch.set_float32_matmul_precision('high')
train.a_device = torch.device('cuda')
model = train.Adder().cuda()
model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
model.eval()
for structured in (0.0, 0.75):
    e, n, ex = exact_errors(model, 128 if structured == 0 else 64, 8192, structured)
    print('random' if not structured else 'structured', e, n, ex)

pairs = edge_pairs()
a = torch.tensor([x for x, _ in pairs], device='cuda')
b = torch.tensor([y for _, y in pairs], device='cuda')
with torch.no_grad():
    pred = model(train.to_digits(a), train.to_digits(b)).argmax(-1)
actual = a + b
pv = torch.tensor([from_digits(x) for x in pred.cpu().tolist()], device='cuda')
bad = (pv != actual).nonzero().flatten()
print('edges', len(bad), len(pairs), [(pairs[i], pv[i].item(), actual[i].item()) for i in bad[:10].tolist()])

# Attention dependence: compare first block's QK scores for different complete inputs.
x1 = model.a_embed(train.to_digits(torch.tensor([123], device='cuda'))) + model.b_embed(train.to_digits(torch.tensor([456], device='cuda'))) + model.pos
x2 = model.a_embed(train.to_digits(torch.tensor([987654321], device='cuda'))) + model.b_embed(train.to_digits(torch.tensor([111111111], device='cuda'))) + model.pos
block = model.blocks[0]
q1, k1, _ = block.qkv(block.n1(x1)).chunk(3, -1)
q2, k2, _ = block.qkv(block.n1(x2)).chunk(3, -1)
print('qk_max_change', ((q1 @ k1.transpose(-1,-2)) - (q2 @ k2.transpose(-1,-2))).abs().max().item())

# Graded module on CPU and parameter sensitivity.
spec = importlib.util.spec_from_file_location('submission', '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
cpu, meta = sub.build_model()
print('submission_parameters', sum(p.numel() for p in cpu.parameters()), meta)
checks = [(0,0), (1,2), (99_999_999_999_999,1), (12_345_678_901_234, 87_654_321_098_765), (99_999_999_999_999,99_999_999_999_999)]
print('cpu_checks', [(a,b,sub.add(cpu,a,b),a+b) for a,b in checks])
zero = copy.deepcopy(cpu)
with torch.no_grad():
    for p in zero.parameters(): p.zero_()
print('zero_parameter_checks', [sub.add(zero,a,b) for a,b in checks])
