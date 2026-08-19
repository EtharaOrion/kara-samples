import copy
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import Model

DEVICE = "cuda"
BATCH = 4096
POW10 = torch.tensor([10**i for i in range(15)], device=DEVICE, dtype=torch.long)
LIMIT = 100_000_000_000_000


def digits(x):
    return (x[:, None] // POW10[None, :]) % 10


def uniform(n):
    return (torch.randint(0, LIMIT, (n,), device=DEVICE),
            torch.randint(0, LIMIT, (n,), device=DEVICE))


def structured(n):
    # Regenerated families: shifted carry/non-carry runs, sparse boundaries,
    # repeated digits, complements, and maximum-overflow contrasts.
    a, b = uniform(n)
    family = torch.randint(0, 8, (n,), device=DEVICE)
    start = torch.randint(0, 14, (n,), device=DEVICE)
    maxlen = 14 - start
    length = 1 + (torch.rand(n, device=DEVICE) * maxlen).long()
    p = POW10[start]
    run = POW10[length] - 1
    idx = family == 0
    # all-9 shifted chain plus sparse increment
    a[idx] = (run[idx] * p[idx]).clamp_max(LIMIT - 1)
    b[idx] = p[idx]
    idx = family == 1
    # matched near-chain that must not propagate
    a[idx] = ((run[idx] - 1).clamp_min(0) * p[idx]).clamp_max(LIMIT - 1)
    b[idx] = p[idx]
    idx = family == 2
    # arbitrary digit pairs isolated at random columns, emphasizing equal boundaries
    da = torch.randint(0, 10, (n,), device=DEVICE)
    db = torch.randint(0, 10, (n,), device=DEVICE)
    force = torch.rand(n, device=DEVICE) < .5
    da[force] = torch.where(torch.rand(n, device=DEVICE)[force] < .5, 5, 9)
    db[force] = da[force]
    a[idx] = da[idx] * p[idx]
    b[idx] = db[idx] * p[idx]
    idx = family == 3
    # repeated decimal digits
    rep = (LIMIT - 1) // 9
    a[idx] = torch.randint(0, 10, (n,), device=DEVICE)[idx] * rep
    b[idx] = torch.randint(0, 10, (n,), device=DEVICE)[idx] * rep
    idx = family == 4
    # random prefix followed by a guaranteed carry run
    lowmask = POW10[start] - 1
    base = torch.randint(0, LIMIT, (n,), device=DEVICE)
    chain = (run * p).clamp_max(LIMIT - 1)
    a[idx] = ((base[idx] // POW10[start[idx] + length[idx]]) * POW10[start[idx] + length[idx]] + chain[idx] + base[idx] % lowmask[idx].clamp_min(1)).clamp_max(LIMIT - 1)
    b[idx] = p[idx]
    idx = family == 5
    # powers of ten against maximum or maximum-minus-one
    a[idx] = LIMIT - 1
    b[idx] = p[idx]
    idx = family == 6
    # complement blocks whose low block sums to a power of ten
    low = 1 + (torch.rand(n, device=DEVICE) * run).long()
    a[idx] = (low[idx] * p[idx]).clamp_max(LIMIT - 1)
    b[idx] = ((POW10[length[idx]] - low[idx]) * p[idx]).clamp_max(LIMIT - 1)
    return a, b


def batch(step):
    if step < 12000:
        a, b = uniform(BATCH)
    else:
        n = BATCH // 2
        a0, b0 = uniform(n)
        a1, b1 = structured(BATCH - n)
        a, b = torch.cat((a0, a1)), torch.cat((b0, b1))
    s = a + b
    return digits(a), digits(b), digits(s)

@torch.no_grad()
def evaluate(model, count, kind="uniform", chunk=16384):
    model.eval()
    errors = 0
    digit_errors = 0
    for _ in range(math.ceil(count / chunk)):
        n = min(chunk, count)
        count -= n
        a, b = uniform(n) if kind == "uniform" else structured(n)
        target = digits(a + b)
        pred = model(digits(a), digits(b)).argmax(-1)
        bad = (pred != target)
        errors += bad.any(1).sum().item()
        digit_errors += bad.sum().item()
    model.train()
    return errors, digit_errors


def export(model):
    source = Path("/workspace/submission.py").read_text()
    marker = "def build_model():"
    prefix = source[:source.index(marker)]
    states = {k: v.detach().float().cpu().tolist() for k, v in model.state_dict().items()}
    tail = '''def build_model():\n    model = Model()\n    state = STATE\n    model.load_state_dict({k: torch.tensor(v) for k, v in state.items()})\n    model.eval()\n    return model, {"architecture": "two-block causal refinement transformer", "digits": 14}\n\n\ndef add(model, a: int, b: int) -> int:\n    device = next(model.parameters()).device\n    ad = torch.tensor([[int(c) for c in f"{a:014d}"[::-1]] + [0]], device=device)\n    bd = torch.tensor([[int(c) for c in f"{b:014d}"[::-1]] + [0]], device=device)\n    with torch.no_grad():\n        digits = model(ad, bd).argmax(-1)[0].tolist()\n    text = "".join(str(d) for d in reversed(digits)).lstrip("0")\n    return int(text or "0")\n'''
    Path("/workspace/submission.py").write_text(prefix + "STATE = " + repr(states) + "\n\n" + tail)


def main():
    torch.manual_seed(45001)
    random.seed(45001)
    torch.set_float32_matmul_precision("high")
    model = Model().to(DEVICE).train()
    params = sum(p.numel() for p in model.parameters())
    print("parameters", params, flush=True)
    print("eager", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(.9, .98), weight_decay=.003)
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    best = None
    best_score = 10**9
    started = time.time()
    for step in range(total):
        if step == 12000:
            for g in opt.param_groups: g["lr"] = 1e-3
        if step == 36000:
            for g in opt.param_groups: g["lr"] = 3e-4
        if step == 68000:
            for g in opt.param_groups: g["lr"] = 1e-4
        if step == 96000:
            for g in opt.param_groups: g["lr"] = 3e-5
        if step == 112000:
            for g in opt.param_groups: g["lr"] = 1e-5
        a, b, y = batch(step)
        logits = model(a, b)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(step + 1, float(loss), opt.param_groups[0]["lr"], "sec", round(time.time()-started), flush=True)
        if (step + 1) >= 36000 and (step + 1) % 8000 == 0:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            eu = evaluate(raw, 131072, "uniform")
            es = evaluate(raw, 131072, "structured")
            score = eu[0] * 4 + es[0]
            print("eval", step + 1, eu, es, "score", score, flush=True)
            if score <= best_score:
                best_score = score
                best = copy.deepcopy(raw.state_dict())
                torch.save(best, "/workspace/best.pt")
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    if best is not None: raw.load_state_dict(best)
    print("final uniform", evaluate(raw, 1048576, "uniform"), flush=True)
    print("final structured", evaluate(raw, 524288, "structured"), flush=True)
    export(raw)

if __name__ == "__main__":
    main()
