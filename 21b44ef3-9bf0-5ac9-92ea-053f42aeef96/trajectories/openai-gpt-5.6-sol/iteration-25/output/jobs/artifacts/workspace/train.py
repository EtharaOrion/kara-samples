import argparse
import importlib.util
import math
import os
import random
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DEVICE = "cuda"
LOW = 10_000_000
HIGH = 100_000_000


def digits(x, count=8):
    places = (10 ** torch.arange(count, device=x.device, dtype=torch.long))
    return (x[:, None] // places) % 10


def make_uniform(n):
    return (torch.randint(LOW, HIGH, (n,), device=DEVICE),
            torch.randint(LOW, HIGH, (n,), device=DEVICE))


def make_structured(n):
    a, b = make_uniform(n)
    family = torch.randint(0, 9, (n,), device=DEVICE)
    # Complements and near-complements around 1e8 and nearby round totals.
    m = family == 0
    aa = torch.randint(LOW, 90_000_001, (n,), device=DEVICE)
    delta = torch.randint(-20, 21, (n,), device=DEVICE)
    bb = 100_000_000 - aa + delta
    valid = m & (bb >= LOW) & (bb < HIGH)
    a[valid], b[valid] = aa[valid], bb[valid]
    # Force asymmetric suffixes of 9s/0s, all carry lengths.
    for fam, digit_a, digit_b in [(1, 9, 1), (2, 0, 9), (3, 9, 0)]:
        m = family == fam
        run = torch.randint(1, 8, (n,), device=DEVICE)
        p = 10 ** run
        prefix_a = torch.randint(1, 10 ** 7, (n,), device=DEVICE)
        prefix_b = torch.randint(1, 10 ** 7, (n,), device=DEVICE)
        suffix_a = p - 1 if digit_a == 9 else torch.zeros_like(p)
        suffix_b = p - 1 if digit_b == 9 else (torch.ones_like(p) if digit_b == 1 else torch.zeros_like(p))
        aa = (prefix_a * p + suffix_a).clamp(LOW, HIGH - 1)
        bb = (prefix_b * p + suffix_b).clamp(LOW, HIGH - 1)
        a[m], b[m] = aa[m], bb[m]
    # Decimal-boundary neighborhoods at every scale.
    m = family == 4
    run = torch.randint(1, 8, (n,), device=DEVICE)
    p = 10 ** run
    center = torch.randint(1, 10 ** 7, (n,), device=DEVICE) * p
    aa = (center + torch.randint(-100, 101, (n,), device=DEVICE)).clamp(LOW, HIGH - 1)
    bb = torch.randint(LOW, HIGH, (n,), device=DEVICE)
    a[m], b[m] = aa[m], bb[m]
    # Repeated digits.
    m = family == 5
    rep = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=DEVICE)
    aa = rep[torch.randint(0, len(rep), (n,), device=DEVICE)]
    bb = rep[torch.randint(0, len(rep), (n,), device=DEVICE)]
    a[m], b[m] = aa[m], bb[m]
    # Sparse decimal operands.
    m = family == 6
    pos1, pos2 = torch.randint(0, 8, (2, n), device=DEVICE)
    aa = 10_000_000 + torch.randint(0, 10, (n,), device=DEVICE) * (10 ** pos1)
    bb = 10_000_000 + torch.randint(0, 10, (n,), device=DEVICE) * (10 ** pos2)
    a[m], b[m] = aa[m].clamp_max(HIGH-1), bb[m].clamp_max(HIGH-1)
    # Near extrema.
    m = family == 7
    aa = torch.where(torch.rand(n, device=DEVICE) < .5, LOW + torch.randint(0, 10000, (n,), device=DEVICE), HIGH - 1 - torch.randint(0, 10000, (n,), device=DEVICE))
    bb = torch.where(torch.rand(n, device=DEVICE) < .5, LOW + torch.randint(0, 10000, (n,), device=DEVICE), HIGH - 1 - torch.randint(0, 10000, (n,), device=DEVICE))
    a[m], b[m] = aa[m], bb[m]
    # Random rounded operands with independently selected scales.
    m = family == 8
    pa = 10 ** torch.randint(1, 8, (n,), device=DEVICE)
    pb = 10 ** torch.randint(1, 8, (n,), device=DEVICE)
    aa = (torch.randint(LOW, HIGH, (n,), device=DEVICE) // pa * pa).clamp(LOW, HIGH-1)
    bb = (torch.randint(LOW, HIGH, (n,), device=DEVICE) // pb * pb).clamp(LOW, HIGH-1)
    a[m], b[m] = aa[m], bb[m]
    return a, b


def batch(n, structured=.35):
    a, b = make_uniform(n)
    m = torch.rand(n, device=DEVICE) < structured
    sa, sb = make_structured(n)
    a[m], b[m] = sa[m], sb[m]
    ad, bd, sd = digits(a), digits(b), digits(a + b, 9)
    operands = torch.stack((ad, bd), 2).reshape(n, 16)
    seq = torch.cat((operands, torch.full((n, 1), 10, device=DEVICE, dtype=torch.long), sd[:, :8]), 1)
    return seq, sd


def autoregressive(model, a, b):
    n = a.numel()
    seq = torch.cat((torch.stack((digits(a), digits(b)), 2).reshape(n, 16),
                     torch.full((n, 1), 10, device=DEVICE, dtype=torch.long)), 1)
    out = []
    for _ in range(9):
        d = model(seq)[:, -1].argmax(1)
        out.append(d)
        seq = torch.cat((seq, d[:, None]), 1)
    return torch.stack(out, 1)


@torch.no_grad()
def validate(model, n=100000, structured=False, chunk=10000):
    model.eval(); good = total = 0; min_margin = 1e9
    for _ in range((n + chunk - 1) // chunk):
        size = min(chunk, n-total)
        a, b = make_structured(size) if structured else make_uniform(size)
        pred = autoregressive(model, a, b)
        truth = digits(a+b, 9)
        good += (pred == truth).all(1).sum().item(); total += size
    model.train()
    return good, total


def prune(model, width):
    old = model.ff_in.out_features
    assert width < old
    score = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
    keep = score.topk(width).indices.sort().values
    new = AdditionTransformer(width).to(DEVICE)
    state = model.state_dict()
    state["ff_in.weight"] = state["ff_in.weight"][keep]
    state["ff_out.weight"] = state["ff_out.weight"][:, keep]
    new.load_state_dict(state)
    return new


def export(model, path="/workspace/submission.py"):
    source = open(path).read()
    start = source.index("_TRAINED_STATE =")
    end = source.index("\n\n\ndef build_model", start)
    parts = ["_TRAINED_STATE = {"]
    for name, tensor in model.state_dict().items():
        vals = tensor.detach().float().cpu().reshape(-1).tolist()
        literal = repr(vals)
        parts.append(f"    {name!r}: torch.tensor({literal}).reshape({tuple(tensor.shape)!r}),")
    parts.append("}")
    source = source[:start] + "\n".join(parts) + source[end:]
    source = source.replace("model = AdditionTransformer(4)", f"model = AdditionTransformer({model.ff_in.out_features})")
    open(path, "w").write(source)


def train_stage(model, steps, lr, batch_size, structured, label, save_every=2000):
    torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.98), weight_decay=.01, fused=True)
    model.train()
    for step in range(1, steps+1):
        seq, target = batch(batch_size, structured)
        logits = model(seq)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1,10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 500 == 0:
            print(label, step, f"loss={loss.item():.6f}", flush=True)
        if step % save_every == 0 or step == steps:
            torch.save({"width": model.ff_in.out_features, "state": model.state_dict()}, f"/workspace/{label}.pt")
            export(model)
    return model


def main():
    p=argparse.ArgumentParser(); p.add_argument("--phase", choices=["teacher","prune3","prune2","prune1"], default="teacher"); p.add_argument("--steps", type=int); args=p.parse_args()
    torch.manual_seed(20250308)
    if args.phase == "teacher":
        model=AdditionTransformer(4).to(DEVICE)
        model=train_stage(model,args.steps or 36000,2e-3,8192,.35,"teacher")
    else:
        src={"prune3":"teacher.pt","prune2":"prune3.pt","prune1":"prune2.pt"}[args.phase]
        ck=torch.load("/workspace/"+src,weights_only=True)
        model=AdditionTransformer(ck["width"]).to(DEVICE); model.load_state_dict(ck["state"])
        target={"prune3":3,"prune2":2,"prune1":1}[args.phase]
        model=prune(model,target)
        defaults={3:22000,2:50000,1:60000}
        model=train_stage(model,args.steps or defaults[target],2e-5 if target>=2 else 1e-5,8192,.5,args.phase)
    r=validate(model,100000,False); s=validate(model,100000,True)
    print("VALID",r,s,flush=True); export(model)

if __name__ == "__main__": main()
