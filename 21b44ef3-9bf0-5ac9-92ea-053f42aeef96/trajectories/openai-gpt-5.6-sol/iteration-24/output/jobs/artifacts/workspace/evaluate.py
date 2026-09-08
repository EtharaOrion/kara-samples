import argparse
import importlib.util
import random
from pathlib import Path
import torch

ROOT = Path('/workspace')
spec = importlib.util.spec_from_file_location('s', ROOT / 'submission.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
model, meta = s.build_model(); model.cuda().eval()

@torch.no_grad()
def decode(a, b):
    n = len(a)
    div = torch.tensor([10**i for i in range(8)], device='cuda')
    ad = (a[:, None] // div) % 10; bd = (b[:, None] // div) % 10
    x = torch.empty((n, 17), dtype=torch.long, device='cuda')
    x[:, 0:16:2] = ad; x[:, 1:16:2] = bd; x[:, 16] = 10
    result = torch.zeros(n, dtype=torch.long, device='cuda')
    place = 1
    for _ in range(9):
        digit = model(x)[:, -1].argmax(-1)
        result += digit * place; place *= 10
        x = torch.cat((x, digit[:, None]), 1)
    return result

def random_eval(total, structured=False):
    correct = 0
    for start in range(0, total, 8192):
        n = min(8192, total-start)
        a = torch.randint(10_000_000, 100_000_000, (n,), device='cuda')
        b = torch.randint(10_000_000, 100_000_000, (n,), device='cuda')
        if structured:
            half=n//2
            target = 100_000_000 + torch.randint(-9999,10000,(half,),device='cuda')
            bb=target-a[:half]
            b[:half]=torch.where((bb>=10_000_000)&(bb<100_000_000),bb,100_000_000-a[:half])
        correct += int((decode(a,b)==a+b).sum())
    print(('structured' if structured else 'random'), correct, total, correct/total)

values = sorted(set([10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_999,
 11_111_111,20_000_000,40_000_000,50_000_000,80_000_000,89_999_999,90_000_000,
 98_999_999,99_000_000,99_000_001,99_899_999,99_900_000,99_990_000,99_999_000,
 99_999_900,99_999_990,99_999_998,99_999_999] +
 [10_000_000 + 10**p + d for p in range(8) for d in (-1,0,1)] +
 [100_000_000 - 10**p + d for p in range(8) for d in (-1,0,1)]))
pairs=[(a,b) for a in values for b in values if 10_000_000<=a<100_000_000 and 10_000_000<=b<100_000_000]
correct=0; failures=[]
for start in range(0,len(pairs),8192):
    chunk=pairs[start:start+8192]
    a=torch.tensor([q[0] for q in chunk],device='cuda'); b=torch.tensor([q[1] for q in chunk],device='cuda')
    out=decode(a,b)
    ok=out==a+b; correct+=int(ok.sum())
    for i in (~ok).nonzero()[:20]:
        j=int(i); failures.append((int(a[j]),int(b[j]),int(out[j]),int(a[j]+b[j])))
print('cartesian',correct,len(pairs),correct/len(pairs),'sample failures',failures[:20])
random_eval(200_000,False); random_eval(200_000,True)
print(meta, sum(p.numel() for p in model.parameters()))
