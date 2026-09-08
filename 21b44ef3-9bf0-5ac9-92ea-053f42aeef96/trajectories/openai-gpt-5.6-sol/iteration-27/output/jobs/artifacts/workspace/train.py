import argparse
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from submission import AdditionTransformer

DEVICE = "cuda"
LOW = 10_000_000
HIGH = 100_000_000
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def sample_pairs(n, structured=0.35):
    a = torch.randint(LOW, HIGH, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH, (n,), device=DEVICE)
    count = int(n * structured)
    if not count:
        return a, b
    mode = torch.randint(0, 8, (count,), device=DEVICE)
    x = torch.randint(LOW, HIGH, (count,), device=DEVICE)
    y = torch.randint(LOW, HIGH, (count,), device=DEVICE)

    # Complements and near-complements exercise carries through every column.
    idx = mode == 0
    delta = torch.randint(-1000, 1001, (count,), device=DEVICE)
    y[idx] = (HIGH - x[idx] + delta[idx]).clamp(LOW, HIGH - 1)

    # Decimal boundaries at all scales.
    idx = mode == 1
    scale = POW10[torch.randint(1, 8, (count,), device=DEVICE)]
    base = (x // scale) * scale
    jitter = torch.randint(-20, 21, (count,), device=DEVICE)
    x[idx] = (base[idx] + jitter[idx]).clamp(LOW, HIGH - 1)

    # Unequal zero/nine suffix lengths and arbitrary leading digits.
    idx = mode == 2
    ka = torch.randint(1, 8, (count,), device=DEVICE)
    kb = torch.randint(1, 8, (count,), device=DEVICE)
    sa, sb = POW10[ka], POW10[kb]
    x9 = (x // sa) * sa + sa - 1
    y0 = (y // sb) * sb + torch.randint(0, 10, (count,), device=DEVICE)
    x[idx] = x9[idx].clamp(LOW, HIGH - 1)
    y[idx] = y0[idx].clamp(LOW, HIGH - 1)

    # Long carry starts at variable positions.
    idx = mode == 3
    k = torch.randint(1, 8, (count,), device=DEVICE)
    scale = POW10[k]
    suffix = scale - 1
    x[idx] = (((x // scale) * scale) + suffix)[idx].clamp(LOW, HIGH - 1)
    small = POW10[torch.randint(0, 7, (count,), device=DEVICE)]
    y[idx] = (((y // scale) * scale) + small)[idx].clamp(LOW, HIGH - 1)

    # Repeated digits.
    idx = mode == 4
    rep = torch.randint(1, 10, (count,), device=DEVICE) * 11_111_111
    x[idx] = rep[idx]
    rep2 = torch.randint(1, 10, (count,), device=DEVICE) * 11_111_111
    y[idx] = rep2[idx]

    # Sparse interior digits with valid nonzero leading digit.
    idx = mode == 5
    lead = torch.randint(1, 10, (count,), device=DEVICE) * 10_000_000
    p1 = POW10[torch.randint(0, 7, (count,), device=DEVICE)]
    p2 = POW10[torch.randint(0, 7, (count,), device=DEVICE)]
    x[idx] = (lead + torch.randint(0, 10, (count,), device=DEVICE) * p1)[idx]
    y[idx] = (torch.randint(1, 10, (count,), device=DEVICE) * 10_000_000 + torch.randint(0, 10, (count,), device=DEVICE) * p2)[idx]

    # Near extrema and broad powers-of-ten neighborhoods.
    idx = mode == 6
    choose = torch.randint(0, 2, (count,), device=DEVICE)
    near = torch.where(choose.bool(), LOW + torch.randint(0, 10000, (count,), device=DEVICE), HIGH - 1 - torch.randint(0, 10000, (count,), device=DEVICE))
    x[idx] = near[idx]

    # Independent rounded prefixes plus offsets, including asymmetric carries.
    idx = mode == 7
    ka = torch.randint(1, 8, (count,), device=DEVICE)
    kb = torch.randint(1, 8, (count,), device=DEVICE)
    sa, sb = POW10[ka], POW10[kb]
    ox = torch.randint(-9, 10, (count,), device=DEVICE)
    oy = torch.randint(-9, 10, (count,), device=DEVICE)
    x[idx] = ((x // sa) * sa + ox)[idx].clamp(LOW, HIGH - 1)
    y[idx] = ((y // sb) * sb + oy)[idx].clamp(LOW, HIGH - 1)

    a[:count], b[:count] = x, y
    perm = torch.randperm(n, device=DEVICE)
    return a[perm], b[perm]


def digits(values, width):
    return ((values[:, None] // POW10[:width]) % 10).long()


def batch(n, structured):
    a, b = sample_pairs(n, structured)
    ad, bd = digits(a, 8), digits(b, 8)
    out = digits(a + b, 9)
    tokens = torch.empty((n, 25), dtype=torch.long, device=DEVICE)
    tokens[:, 0:16:2] = ad
    tokens[:, 1:16:2] = bd
    tokens[:, 16] = 10
    tokens[:, 17:] = out[:, :-1]
    return tokens, out


@torch.no_grad()
def evaluate(model, n=100000, structured=0.0, batch_size=10000, seed=12345):
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    correct = total = 0
    min_margin = 1e9
    model.eval()
    for start in range(0, n, batch_size):
        size = min(batch_size, n - start)
        a, b = sample_pairs(size, structured)
        ad, bd = digits(a, 8), digits(b, 8)
        expected = digits(a + b, 9)
        seq = torch.empty((size, 17), dtype=torch.long, device=DEVICE)
        seq[:, 0:16:2], seq[:, 1:16:2], seq[:, 16] = ad, bd, 10
        predicted = []
        margins = []
        for _ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, dim=-1).values
            margins.append(top[:, 0] - top[:, 1])
            d = logits.argmax(-1)
            predicted.append(d)
            seq = torch.cat((seq, d[:, None]), 1)
        pred = torch.stack(predicted, 1)
        correct += (pred == expected).all(1).sum().item()
        total += size
        min_margin = min(min_margin, torch.stack(margins, 1).min().item())
    torch.random.set_rng_state(state)
    torch.cuda.set_rng_state(cuda_state)
    return correct, total, min_margin


def load_model(path):
    item = torch.load(path, map_location=DEVICE, weights_only=True)
    state = item.get("state", item)
    width = item.get("ff_width", state["ff_in.weight"].shape[0])
    rank = item.get("pos_rank", state["pos_left"].shape[1])
    model = AdditionTransformer(width, rank).to(DEVICE)
    model.load_state_dict(state)
    return model


def save_model(model, path, step=0):
    torch.save({"ff_width": model.ff_in.out_features, "pos_rank": model.pos_left.shape[1], "step": step, "state": model.state_dict()}, path)


def prune(source, destination):
    old = load_model(source)
    width = old.ff_in.out_features
    score = old.ff_in.weight.norm(dim=1) * old.ff_out.weight.norm(dim=0)
    keep = score.topk(width - 1).indices.sort().values
    new = AdditionTransformer(width - 1, old.pos_left.shape[1]).to(DEVICE)
    state = old.state_dict()
    state["ff_in.weight"] = state["ff_in.weight"][keep]
    state["ff_out.weight"] = state["ff_out.weight"][:, keep]
    new.load_state_dict(state)
    save_model(new, destination)
    print(f"pruned {width}->{width-1}; removed score {score.min().item():.6g}")


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model = load_model(args.resume) if args.resume else AdditionTransformer(args.width, args.pos_rank).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    best_score = -1
    for step in range(1, args.steps + 1):
        if args.cosine:
            warm = min(1.0, step / max(1, args.warmup))
            decay = 0.5 * (1 + math.cos(math.pi * step / args.steps))
            lr = args.lr * warm * (0.05 + 0.95 * decay)
            for group in optimizer.param_groups:
                group["lr"] = lr
        tokens, target = batch(args.batch_size, args.structured)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tokens)[:, 16:25]
            per_token = F.cross_entropy(
                logits.reshape(-1, 10), target.reshape(-1), reduction="none"
            ).view(-1, 9)
            weights = torch.ones(9, device=DEVICE)
            weights[args.focus_position] = args.focus_weight
            loss = (per_token * weights).sum() / (weights.sum() * tokens.shape[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                token_acc = (logits.argmax(-1) == target).float().mean().item()
            print(f"step={step} loss={loss.item():.6f} token={token_acc:.6f} lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            r = evaluate(model, args.eval_size, 0.0, seed=100000 + step)
            s = evaluate(model, args.eval_size, 0.8, seed=200000 + step)
            score = min(r[0] / r[1], s[0] / s[1])
            print(f"eval step={step} random={r[0]}/{r[1]} structured={s[0]}/{s[1]} margins={r[2]:.3f},{s[2]:.3f}", flush=True)
            if score >= best_score:
                best_score = score
                save_model(model, args.output + ".best", step)
            model.train()
    save_model(model, args.output, args.steps)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--width", type=int, default=4)
    t.add_argument("--pos-rank", type=int, default=3)
    t.add_argument("--resume")
    t.add_argument("--output", required=True)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--batch-size", type=int, default=8192)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--structured", type=float, default=0.35)
    t.add_argument("--seed", type=int, default=27)
    t.add_argument("--cosine", action="store_true")
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument("--eval-every", type=int, default=4000)
    t.add_argument("--eval-size", type=int, default=20000)
    t.add_argument("--focus-position", type=int, default=0)
    t.add_argument("--focus-weight", type=float, default=1.0)
    q = sub.add_parser("prune")
    q.add_argument("source")
    q.add_argument("destination")
    e = sub.add_parser("eval")
    e.add_argument("checkpoint")
    e.add_argument("--size", type=int, default=100000)
    e.add_argument("--structured", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "prune":
        prune(args.source, args.destination)
    else:
        model = load_model(args.checkpoint)
        result = evaluate(model, args.size, args.structured, seed=args.seed)
        print(result, result[0] / result[1])


if __name__ == "__main__":
    main()
