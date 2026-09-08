import importlib.util
import random
import torch

spec = importlib.util.spec_from_file_location('graded_submission', '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
model, metadata = sub.build_model()
model.cuda()


def encode(a, b):
    places = torch.tensor([10 ** i for i in range(8)], device='cuda')
    ad, bd = a[:, None] // places % 10, b[:, None] // places % 10
    x = torch.empty(a.numel(), 17, dtype=torch.long, device='cuda')
    x[:, 0:16:2], x[:, 1:16:2], x[:, 16] = ad, bd, 10
    return x


@torch.no_grad()
def autoregressive(a, b):
    x = encode(a, b)
    out = torch.zeros_like(a)
    place = 1
    for step in range(9):
        digit = model(x)[:, -1].argmax(-1)
        out += digit * place
        place *= 10
        if step < 8: x = torch.cat((x, digit[:, None]), 1)
    return out


def check(a, b, label):
    pred = autoregressive(a, b)
    bad = pred != a + b
    print(label, int((~bad).sum()), '/', a.numel())
    if bad.any():
        ids = bad.nonzero()[:10, 0]
        print('failures', list(zip(a[ids].tolist(), b[ids].tolist(), pred[ids].tolist(), (a+b)[ids].tolist())))
    return int(bad.sum())


torch.manual_seed(918273)
n = 500000
a = torch.randint(10_000_000, 100_000_000, (n,), device='cuda')
b = torch.randint(10_000_000, 100_000_000, (n,), device='cuda')
bad = check(a, b, 'uniform')

# Structured asymmetric suffixes and complements, generated only for validation.
n = 300000
a = torch.randint(10_000_000, 100_000_000, (n,), device='cuda')
run = torch.randint(1, 8, (n,), device='cuda')
pow10 = 10 ** run
b = (100_000_000 - a + torch.randint(-100, 101, (n,), device='cuda')).clamp(10_000_000, 99_999_999)
a[:n//2] = (a[:n//2] // pow10[:n//2] * pow10[:n//2] + pow10[:n//2] - 1).clamp(10_000_000,99_999_999)
bad += check(a, b, 'structured')

vals = set([10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,
            11_111_111,20_000_000,40_000_009,49_999_999,50_000_000,50_000_001,89_999_999,90_000_000,
            98_999_999,99_000_000,99_899_999,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999])
for p in range(1,8):
    q=10**p
    for lead in range(1,10):
        for d in (-1,0,1):
            v=lead*10_000_000 + q + d
            if 10_000_000 <= v <= 99_999_999: vals.add(v)
vals=sorted(vals)
aa=torch.tensor([x for x in vals for y in vals],device='cuda')
bb=torch.tensor([y for x in vals for y in vals],device='cuda')
bad += check(aa,bb,'edge-grid')

print('metadata',metadata,'parameters',sum(p.numel() for p in model.parameters()),'total_bad',bad)

# Direct public API test on CPU.
model.cpu()
random.seed(4455)
for _ in range(1000):
    x=random.randint(10_000_000,99_999_999); y=random.randint(10_000_000,99_999_999)
    assert sub.add(model,x,y)==x+y
print('direct-api 1000 / 1000')
