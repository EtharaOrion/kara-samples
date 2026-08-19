import sys
import torch
sys.path.insert(0, "/workspace")
from train import Adder, evaluate, result_digits, N

DEVICE = "cuda"
model = Adder(10, 8).to(DEVICE)
model.load_state_dict(torch.load("/workspace/model_h8.pt", weights_only=True))
model.eval()
print("params", sum(p.numel() for p in model.parameters()))
print("uniform", evaluate(model, 128, 8192, False))
print("structured", evaluate(model, 128, 8192, True))

# Systematic decimal boundary, carry length/start, repeated, sparse, and swap cases.
pairs = set()
M = 99_999_999_999_999
base_values = [0, 1, 2, 5, 8, 9, 10, 11, 19, 20, 89, 90, 98, 99, 100, 101, M, M-1, M-8, M-9, 10**13]
for x in base_values:
    for y in base_values:
        if x <= M and y <= M:
            pairs.add((x, y))
for p in range(14):
    unit = 10 ** p
    for length in range(1, 15-p):
        run = (10 ** length - 1) * unit
        for delta in [-2, -1, 0, 1, 2]:
            x = run + delta * unit
            for y in [1, unit, 2*unit, 9*unit, M-run if M >= run else 0]:
                if 0 <= x <= M and 0 <= y <= M:
                    pairs.add((x,y)); pairs.add((y,x))
for digit in range(10):
    x = int(str(digit) * 14)
    for y in [0,1,9,10,11,x,M-x]:
        pairs.add((x,y)); pairs.add((y,x))

def digits(values):
    out = torch.zeros((len(values), N), dtype=torch.long, device=DEVICE)
    work = torch.tensor(values, dtype=torch.long, device=DEVICE)
    for p in range(N):
        out[:,p] = work.remainder(10)
        work.div_(10, rounding_mode="floor")
    return out

pairs = list(pairs)
failures = []
for offset in range(0, len(pairs), 8192):
    part = pairs[offset:offset+8192]
    a = digits([x for x,_ in part]); b = digits([y for _,y in part])
    target = result_digits(a,b)
    pred = model(a,b).argmax(-1)
    bad = pred.ne(target).any(1).nonzero().flatten().tolist()
    failures.extend((part[i], pred[i].tolist(), target[i].tolist()) for i in bad)
print("systematic", len(failures), "/", len(pairs))
print("failure_samples", failures[:20])

# Attention content dependence in the first block.
a = digits([0, 98765432101234]); b = digits([0, 1234567898765])
x = model.a_embed(a) + model.b_embed(b) + model.pos
q,k,_ = model.blocks[0].qkv(model.blocks[0].ln1(x)).chunk(3,-1)
scores = q @ k.transpose(-2,-1)
print("qk_input_difference", (scores[0]-scores[1]).abs().max().item())
