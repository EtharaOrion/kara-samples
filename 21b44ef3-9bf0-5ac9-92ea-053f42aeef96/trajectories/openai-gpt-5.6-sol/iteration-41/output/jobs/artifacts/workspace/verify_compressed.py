import importlib.util
import itertools
import random
from pathlib import Path

import torch

from train import accuracy, encode

spec = importlib.util.spec_from_file_location('candidate', '/workspace/submission_compressed.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model, metadata = module.build_model()
model.cuda().eval()
print('metadata', metadata)
print('parameters', sum(p.numel() for p in model.parameters()))
print('uniform', accuracy(model, 500000, False, 10000))
print('structured', accuracy(model, 500000, True, 10000))

values = {10_000_000, 10_000_001, 10_000_009, 10_000_010, 10_000_099,
          10_000_100, 10_000_999, 10_001_000, 10_009_999, 10_010_000,
          10_099_999, 10_100_000, 10_999_999, 11_000_000, 19_999_999,
          20_000_000, 40_000_009, 49_999_999, 50_000_000, 50_000_001,
          89_999_999, 90_000_000, 98_999_999, 99_000_000, 99_000_001,
          99_899_999, 99_900_000, 99_990_000, 99_999_000, 99_999_900,
          99_999_990, 99_999_998, 99_999_999}
for lead in range(1, 10):
    for k in range(1, 8):
        p = 10 ** k
        for off in (-1, 0, 1, 9):
            x = lead * 10_000_000 + p + off
            if 10_000_000 <= x <= 99_999_999:
                values.add(x)
vals = sorted(values)
pairs = list(itertools.product(vals, vals))
good = 0
with torch.no_grad():
    for start in range(0, len(pairs), 4096):
        part = pairs[start:start+4096]
        a = torch.tensor([x for x, _ in part], device='cuda')
        b = torch.tensor([y for _, y in part], device='cuda')
        x, target = encode(a, b)
        tokens = x[:, :17]
        pred = []
        for _ in range(9):
            digit = model(tokens)[:, -1].argmax(-1)
            pred.append(digit)
            tokens = torch.cat((tokens, digit[:, None]), 1)
        pred = torch.stack(pred, 1)
        good += int((pred == target).all(1).sum())
print('edge', good, len(pairs))

torch.set_num_threads(1)
model.cpu()
random.seed(44)
direct = [(random.randint(10_000_000,99_999_999), random.randint(10_000_000,99_999_999)) for _ in range(2000)]
direct += [(a,b) for a,b in pairs[::max(1,len(pairs)//1000)]]
good = sum(module.add(model,a,b) == a+b for a,b in direct)
print('direct_cpu', good, len(direct))

samples = direct[:40]
normal = [module.add(model,a,b) for a,b in samples]
with torch.no_grad():
    saved = model.o_free.clone()
    model.o_free.zero_()
ablated = [module.add(model,a,b) for a,b in samples]
with torch.no_grad(): model.o_free.copy_(saved)
print('attention_ablation_changes', sum(x != y for x,y in zip(normal,ablated)), len(samples))
with torch.no_grad():
    saved = model.classifier_free.clone()
    model.classifier_free.zero_()
corrupt = [module.add(model,a,b) for a,b in samples]
with torch.no_grad(): model.classifier_free.copy_(saved)
print('classifier_corruption_changes', sum(x != y for x,y in zip(normal,corrupt)), len(samples))
