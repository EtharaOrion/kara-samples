import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from submission import AdditionTransformer

DEVICE = "cuda"
BASE = 100_000_000_000_000
POW10 = torch.tensor([10**i for i in range(15)], device=DEVICE, dtype=torch.long)
POS14 = torch.arange(14, device=DEVICE)


def to_digits(x, n=14):
    return (x[:, None] // POW10[None, :n]) % 10


def from_digits(x):
    return (x * POW10[:x.shape[1]]).sum(1)


def split_sums(s):
    lo = (s - 9).clamp_min(0)
    hi = s.clamp_max(9)
    a = lo + (torch.rand_like(s, dtype=torch.float32) * (hi - lo + 1)).long()
    return a, s - a


def make_batch(batch):
    # Every label is generated from a complete operand pair; families only shape
    # the operand distribution and never supply local transition labels.
    nr = batch // 2
    a0 = torch.randint(BASE, (nr,), device=DEVICE)
    b0 = torch.randint(BASE, (nr,), device=DEVICE)
    chunks_a = [to_digits(a0)]
    chunks_b = [to_digits(b0)]

    # Randomized carry chains, including chains reaching the most significant column.
    n = batch // 4
    ad = torch.randint(10, (n, 14), device=DEVICE)
    bd = torch.randint(10, (n, 14), device=DEVICE)
    start = torch.randint(14, (n,), device=DEVICE)
    max_len = 14 - start
    length = 1 + (torch.rand(n, device=DEVICE) * max_len).long()
    end = start + length
    p = POS14[None, :]
    sums = ad + bd
    start_mask = p == start[:, None]
    body_mask = (p > start[:, None]) & (p < end[:, None])
    stop_mask = (p == end[:, None]) & (end[:, None] < 14)
    start_sum = torch.randint(10, 19, (n, 1), device=DEVICE)
    sums = torch.where(start_mask, start_sum, sums)
    sums = torch.where(body_mask, torch.full_like(sums, 9), sums)
    sums = torch.where(stop_mask, torch.randint(9, (n, 1), device=DEVICE), sums)
    aa, bb = split_sums(sums)
    changed = start_mask | body_mask | stop_mask
    chunks_a.append(torch.where(changed, aa, ad))
    chunks_b.append(torch.where(changed, bb, bd))

    # Non-carry 9-runs contrast visually similar carry chains.
    n = batch // 8
    ad = torch.randint(10, (n, 14), device=DEVICE)
    bd = torch.randint(10, (n, 14), device=DEVICE)
    start = torch.randint(14, (n,), device=DEVICE)
    length = 1 + (torch.rand(n, device=DEVICE) * (14 - start)).long()
    end = start + length
    p = POS14[None, :]
    run = (p >= start[:, None]) & (p < end[:, None])
    before = (p + 1 == start[:, None]) & (start[:, None] > 0)
    sums = torch.where(run, torch.full_like(ad, 9), ad + bd)
    sums = torch.where(before, torch.randint(9, (n, 1), device=DEVICE), sums)
    aa, bb = split_sums(sums)
    changed = run | before
    chunks_a.append(torch.where(changed, aa, ad))
    chunks_b.append(torch.where(changed, bb, bd))

    # Sparse exact-boundary contrasts. Half are the historically difficult 5+5.
    n = batch - sum(x.shape[0] for x in chunks_a)
    ad = torch.zeros((n, 14), dtype=torch.long, device=DEVICE)
    bd = torch.zeros((n, 14), dtype=torch.long, device=DEVICE)
    col = torch.randint(14, (n,), device=DEVICE)
    rows = torch.arange(n, device=DEVICE)
    half = rows < n // 2
    ad[rows, col] = torch.where(half, torch.full_like(rows, 5), torch.randint(1, 10, (n,), device=DEVICE))
    bd[rows, col] = torch.where(half, torch.full_like(rows, 5), 10 - ad[rows, col])
    extra = torch.randint(14, (n,), device=DEVICE)
    bd[rows, extra] = torch.where(extra == col, bd[rows, extra], torch.randint(10, (n,), device=DEVICE))
    chunks_a.append(ad)
    chunks_b.append(bd)

    ad = torch.cat(chunks_a)
    bd = torch.cat(chunks_b)
    order = torch.randperm(batch, device=DEVICE)
    ad, bd = ad[order], bd[order]
    total = from_digits(ad) + from_digits(bd)
    target = to_digits(total, 15)
    return ad, bd, target


def all_logits(model, ad, bd, target):
    batch = ad.shape[0]
    pad = torch.full((batch, 1), 10, dtype=torch.long, device=ad.device)
    source_a = torch.cat((ad, pad), 1)
    source_b = torch.cat((bd, pad), 1)
    x = model.a_embed(source_a) + model.b_embed(source_b)
    x = torch.cat((x, model.out_embed(target[:, :-1])), 1)
    x = x + model.position[None, :29]
    for block in model.blocks:
        x = block(x)
    return model.head(model.final_norm(x[:, 14:]))


@torch.no_grad()
def decode(model, ad, bd):
    generated = torch.empty((ad.shape[0], 0), dtype=torch.long, device=ad.device)
    for _ in range(15):
        generated = torch.cat((generated, model(ad, bd, generated).argmax(1, keepdim=True)), 1)
    return generated


@torch.no_grad()
def random_errors(model, count, batch=16384):
    errors = 0
    for _ in range(math.ceil(count / batch)):
        n = min(batch, count)
        count -= n
        a = torch.randint(BASE, (n,), device=DEVICE)
        b = torch.randint(BASE, (n,), device=DEVICE)
        ad, bd = to_digits(a), to_digits(b)
        errors += (decode(model, ad, bd) != to_digits(a + b, 15)).any(1).sum().item()
    return errors


@torch.no_grad()
def structured_errors(model, count, batch=16384):
    errors = 0
    for _ in range(math.ceil(count / batch)):
        n = min(batch, count)
        count -= n
        ad, bd, target = make_batch(n)
        errors += (decode(model, ad, bd) != target).any(1).sum().item()
    return errors


def systematic_pairs():
    pairs = {(0, 0), (BASE - 1, 1), (BASE - 1, BASE - 1), (99_999_999_999_99, 1)}
    for k in range(14):
        p = 10**k
        for x in range(1, 10):
            pairs.add((x, p + (10 - x)))
            pairs.add((p - 1 if p > 1 else 9, 1))
            pairs.add((p * x, p * (10 - x)))
        for run in range(1, 15 - k):
            r = (10**run - 1) * p
            if r < BASE:
                pairs.add((r, p))
                pairs.add((r, 1))
                pairs.add((r, r))
    pairs = [(a, b) for a, b in pairs if 0 <= a < BASE and 0 <= b < BASE]
    return pairs


@torch.no_grad()
def systematic_errors(model):
    pairs = systematic_pairs()
    a = torch.tensor([x[0] for x in pairs], device=DEVICE)
    b = torch.tensor([x[1] for x in pairs], device=DEVICE)
    wrong = (decode(model, to_digits(a), to_digits(b)) != to_digits(a + b, 15)).any(1)
    bad = [(pairs[i], int(a[i] + b[i])) for i in wrong.nonzero().flatten().tolist()[:20]]
    return int(wrong.sum()), len(pairs), bad


def export_submission(model, path):
    state = {k: v.detach().float().cpu().reshape(-1).tolist() for k, v in model.state_dict().items()}
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    weights = repr(state)
    shape_text = repr(shapes)
    template = Path(__file__).with_name("submission.py").read_text()
    prefix = template.split("\n_WEIGHTS =", 1)[0]
    load_code = "\n_WEIGHTS = " + weights + "\n_SHAPES = " + shape_text + "\n\ndef build_model():\n    model = AdditionTransformer()\n    state = {k: torch.tensor(v, dtype=torch.float32).reshape(_SHAPES[k]) for k, v in _WEIGHTS.items()}\n    model.load_state_dict(state)\n    return model, {\"architecture\": \"aligned autoregressive transformer\", \"digits\": 14, \"parameters\": 2244}\n"
    add_part = template.rsplit("\ndef add(model", 1)[1]
    Path(path).write_text(prefix + load_code + "\ndef add(model" + add_part)


def main():
    torch.manual_seed(2701)
    random.seed(2701)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.005, fused=True)
    steps = 26000
    batch = 4096
    started = time.time()
    best_score = 10**9
    for step in range(1, steps + 1):
        if step == 9001:
            for g in optimizer.param_groups: g["lr"] = 1e-3
        elif step == 15001:
            for g in optimizer.param_groups: g["lr"] = 3e-4
        elif step == 21001:
            for g in optimizer.param_groups: g["lr"] = 1e-4
        ad, bd, target = make_batch(batch)
        logits = all_logits(model, ad, bd, target)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == 1:
            model.eval()
            re = random_errors(model, 32768)
            se = structured_errors(model, 32768)
            sy, total, bad = systematic_errors(model)
            score = re * 10 + se * 5 + sy
            print(f"step {step} loss {loss.item():.6g} random {re}/32768 structured {se}/32768 systematic {sy}/{total} lr {optimizer.param_groups[0]['lr']:.1g} mins {(time.time()-started)/60:.1f}", flush=True)
            if score <= best_score:
                best_score = score
                torch.save(model.state_dict(), Path(__file__).with_name("best.pt"))
                export_submission(model, Path(__file__).with_name("submission.py"))
            if bad: print(" bad", bad[:3], flush=True)
            model.train()
    model.load_state_dict(torch.load(Path(__file__).with_name("best.pt"), weights_only=True))
    model.eval()
    print("FINAL random", random_errors(model, 1048576), flush=True)
    print("FINAL structured", structured_errors(model, 1048576), flush=True)
    print("FINAL systematic", systematic_errors(model), flush=True)
    export_submission(model, Path(__file__).with_name("submission.py"))


if __name__ == "__main__":
    main()
