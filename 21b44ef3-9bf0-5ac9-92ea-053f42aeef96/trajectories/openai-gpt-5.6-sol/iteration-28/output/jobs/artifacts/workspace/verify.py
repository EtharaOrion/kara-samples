import importlib.util
import random
import sys
import torch
sys.path.insert(0, '/workspace')
import submission
from train import TrainModel, evaluate

model, metadata = submission.build_model()
print('parameters', sum(p.numel() for p in model.parameters()), metadata)
trained = TrainModel(2)
trained.load_state_dict(torch.load('/workspace/width2.pt', weights_only=True))
max_delta = max((model.state_dict()[k] - v).abs().max().item() for k, v in trained.state_dict().items())
print('export max weight delta', max_delta)
trained.cuda()
print('random 500k', evaluate(trained, 500000, 0.0))
print('mixed 500k', evaluate(trained, 500000, 0.6))

random.seed(28)
cases = [(random.randrange(10_000_000,100_000_000), random.randrange(10_000_000,100_000_000)) for _ in range(3000)]
edges = [10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_111_111,40_000_009,49_999_999,50_000_000,50_000_001,88_888_888,89_999_999,90_000_000,90_000_001,98_999_999,99_000_000,99_000_001,99_899_999,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999]
cases += [(a,b) for a in edges for b in edges]
correct = 0
failures = []
for a,b in cases:
    got = submission.add(model,a,b)
    if got == a+b: correct += 1
    elif len(failures)<10: failures.append((a,b,got,a+b))
print('direct CPU', correct, '/', len(cases), failures)

sample = cases[:40]
base = [submission.add(model,a,b) for a,b in sample]
with torch.no_grad():
    saved = [layer.weight.clone() for layer in model.attn_out]
    for layer in model.attn_out: layer.weight.zero_()
ablated = [submission.add(model,a,b) for a,b in sample]
with torch.no_grad():
    for layer, weight in zip(model.attn_out,saved): layer.weight.copy_(weight)
    out_saved = model.output.weight.clone()
    model.output.weight.zero_()
corrupt = [submission.add(model,a,b) for a,b in sample]
with torch.no_grad(): model.output.weight.copy_(out_saved)
print('attention changed', sum(x!=y for x,y in zip(base,ablated)), '/40')
print('output changed', sum(x!=y for x,y in zip(base,corrupt)), '/40')
