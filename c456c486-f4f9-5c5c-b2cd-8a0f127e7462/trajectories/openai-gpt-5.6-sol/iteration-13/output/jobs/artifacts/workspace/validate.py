import random
import time

import torch

from submission import AdditionTransformer
from train import evaluate


LIMIT = 10**14 - 1


def edge_cases():
    cases = {(0, 0), (LIMIT, 0), (0, LIMIT), (LIMIT, LIMIT), (LIMIT, 1), (1, LIMIT)}
    for k in range(1, 15):
        p = 10**k
        values = [p - 1, p, p + 1, max(0, p - 10), min(LIMIT, p + 10)]
        for x in values:
            for y in [0, 1, 9, p - 1, min(LIMIT, p), LIMIT - x]:
                if 0 <= x <= LIMIT and 0 <= y <= LIMIT:
                    cases.add((x, y))
                    cases.add((y, x))
        # Every carry length, embedded at multiple boundaries.
        for start in range(15 - k):
            run = (p - 1) * 10**start
            for prefix in [0, 1, 37, 999]:
                x = min(LIMIT, run + prefix * 10 ** (start + k))
                y = 10**start
                cases.add((x, y)); cases.add((y, x))
    patterns = [int(c * 14) for c in "123456789"]
    patterns += [int("09" * 7), int("90" * 7), int("1234567" * 2), int("9876543" * 2)]
    for x in patterns:
        for y in patterns + [1, 9, LIMIT - x]:
            cases.add((x, y)); cases.add((y, x))
    return sorted(cases)


def encode(values, device):
    x = torch.tensor(values, device=device, dtype=torch.long)
    powers = (10 ** torch.arange(15, device=device, dtype=torch.long))[None]
    return x[:, None].div(powers, rounding_mode="floor").remainder(10)


model = AdditionTransformer()
model.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
model.eval()
cases = edge_cases()
for device in ["cuda", "cpu"]:
    model.to(device)
    wrong = []
    started = time.time()
    with torch.no_grad():
        for i in range(0, len(cases), 4096):
            part = cases[i:i+4096]
            a = encode([x for x, _ in part], device)
            b = encode([y for _, y in part], device)
            pred = model(a, b).argmax(-1)
            powers = (10 ** torch.arange(15, device=device, dtype=torch.long))[None]
            got = (pred * powers).sum(1).cpu().tolist()
            for pair, result in zip(part, got):
                if result != pair[0] + pair[1]:
                    wrong.append((pair, result, pair[0] + pair[1]))
    print(device, "cases", len(cases), "wrong", len(wrong), wrong[:10], "seconds", time.time()-started)

# Confirm attention logits genuinely vary with content.
model.cuda()
a = encode([12345678901234, 99999999999999], "cuda")
b = encode([76543210987654, 1], "cuda")
with torch.no_grad():
    x = model.digit(a) + model.digit(b) + model.position
    z = model.norm1(x)
    q, k, _ = model.qkv(z).chunk(3, -1)
    q = q.view(2, 15, 2, 6).transpose(1, 2)
    k = k.view(2, 15, 2, 6).transpose(1, 2)
    scores = q @ k.transpose(-2, -1)
print("qk input delta", (scores[0] - scores[1]).abs().max().item())
