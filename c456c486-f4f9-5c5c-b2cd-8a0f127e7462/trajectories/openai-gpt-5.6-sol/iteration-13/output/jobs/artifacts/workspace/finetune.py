import time

import torch
import torch.nn.functional as F

from submission import AdditionTransformer
from train import random_batch, structured_batch, targets, evaluate


DEVICE = "cuda"
BATCH = 4096


def carry_batch(n, generator):
    a = torch.zeros((n, 15), device=DEVICE, dtype=torch.long)
    b = torch.zeros_like(a)
    start = torch.randint(0, 14, (n,), device=DEVICE, generator=generator)
    length = torch.randint(1, 15, (n,), device=DEVICE, generator=generator)
    end = torch.minimum(start + length, torch.full_like(start, 14))
    # Add optional random context above the carry run, never below it.
    context = torch.randint(10, (n, 15), device=DEVICE, generator=generator)
    for p in range(14):
        above = p > end
        a[:, p] = torch.where(above, context[:, p], a[:, p])
        active = (start <= p) & (p <= end)
        seed = start == p
        av = torch.randint(1, 10, (n,), device=DEVICE, generator=generator)
        bv = torch.where(seed, 10 - av, 9 - av)
        a[:, p] = torch.where(active, av, a[:, p])
        b[:, p] = torch.where(active, bv, b[:, p])
    # Include both sparse increment orientation and varied seed splits.
    sparse = torch.rand(n, device=DEVICE, generator=generator) < .5
    rows = torch.arange(n, device=DEVICE)
    a[rows[sparse], start[sparse]] = 9
    b[rows[sparse], start[sparse]] = 1
    swap = torch.rand(n, device=DEVICE, generator=generator) < .5
    old_a = a.clone()
    a[swap] = b[swap]
    b[swap] = old_a[swap]
    return a, b, targets(a, b)


@torch.no_grad()
def carry_eval(model, batches, seed):
    model.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    wrong = digit_wrong = 0
    for _ in range(batches):
        a, b, y = carry_batch(4096, g)
        pred = model(a, b).argmax(-1)
        bad = pred != y
        wrong += bad.any(1).sum().item()
        digit_wrong += bad.sum().item()
    model.train()
    return wrong, batches * 4096, digit_wrong


torch.manual_seed(1414)
torch.set_float32_matmul_precision("high")
raw = AdditionTransformer().to(DEVICE)
raw.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
model = torch.compile(raw)
optimizer = torch.optim.AdamW(raw.parameters(), lr=5e-5, weight_decay=.001, fused=True)
g = torch.Generator(device=DEVICE).manual_seed(1415)
started = time.time()
best_score = 10**9
for step in range(1, 8001):
    optimizer.param_groups[0]["lr"] = 5e-5 if step <= 4000 else 2e-5
    ra, rb, ry = random_batch(1536, g)
    sa, sb, sy = structured_batch(1024, g)
    ca, cb, cy = carry_batch(1536, g)
    a = torch.cat((ra, sa, ca)); b = torch.cat((rb, sb, cb)); y = torch.cat((ry, sy, cy))
    loss = F.cross_entropy(model(a, b).flatten(0, 1), y.flatten())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    optimizer.step()
    if step % 500 == 0:
        r = evaluate(model, 16, 4096, False, 300000 + step)
        s = evaluate(model, 16, 4096, True, 400000 + step)
        c = carry_eval(model, 16, 500000 + step)
        score = r[0] + s[0] + 4*c[0]
        print(step, loss.item(), "random", r, "structured", s, "carry", c, "score", score, "elapsed", time.time()-started, flush=True)
        if score <= best_score:
            best_score = score
            torch.save(raw.state_dict(), "/workspace/model.pt")
            print("saved", flush=True)

raw.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
print("large random", evaluate(raw, 256, 4096, False, 771414))
print("large structured", evaluate(raw, 128, 4096, True, 881414))
print("large carry", carry_eval(raw, 128, 991414))
