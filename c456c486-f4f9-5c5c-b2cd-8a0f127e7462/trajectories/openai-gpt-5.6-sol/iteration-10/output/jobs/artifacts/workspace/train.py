import sys
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_digits() -> int:
        return 4300
    def _set_digits(maxdigits: int) -> None:
        return None
    sys.get_int_max_str_digits = _get_digits
    sys.set_int_max_str_digits = _set_digits
import argparse
import ast
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdderTransformer

T = 15
MAX_DIGITS = 14


def labels_for(a, b):
    result = torch.empty_like(a)
    carry = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    for p in range(T):
        total = a[:, p] + b[:, p] + carry
        result[:, p] = total.remainder(10)
        carry = total.div(10, rounding_mode="floor")
    return result


def split_sums(sums):
    low = (sums - 9).clamp_min(0)
    span = torch.minimum(sums, torch.full_like(sums, 9)) - low + 1
    a = low + (torch.rand_like(sums, dtype=torch.float) * span).long()
    return a, sums - a


def make_batch(batch, structured_fraction, device):
    a = torch.randint(0, 10, (batch, T), device=device)
    b = torch.randint(0, 10, (batch, T), device=device)
    a[:, -1] = 0
    b[:, -1] = 0
    count = int(batch * structured_fraction)
    if count:
        starts = torch.randint(0, MAX_DIGITS, (count, 1), device=device)
        max_lengths = MAX_DIGITS - starts
        lengths = (torch.rand((count, 1), device=device) * max_lengths).long() + 1
        end = starts + lengths
        pos = torch.arange(T, device=device).view(1, T)
        sums = torch.randint(0, 19, (count, T), device=device)
        sums[:, -1] = 0
        start_mask = pos == starts
        continuation = (pos > starts) & (pos < end)
        before = (starts > 0) & (pos == starts - 1)
        termination = (end < MAX_DIGITS) & (pos == end)
        sums[start_mask] = torch.randint(10, 19, (count,), device=device)
        sums[continuation] = 9
        stop_mask = before | termination
        sums[stop_mask] = torch.randint(0, 9, (int(stop_mask.sum()),), device=device)
        sa, sb = split_sums(sums)
        sa[:, -1] = 0
        sb[:, -1] = 0
        a[:count], b[:count] = sa, sb
    # Complete sparse carry chains and complementary/repeated patterns prevent
    # the leading-zero distribution shift that uniform random data cannot cover.
    special = count // 2
    if special:
        idx = torch.arange(special, device=device)
        starts = torch.randint(0, MAX_DIGITS, (special,), device=device)
        lengths = (torch.rand(special, device=device) * (MAX_DIGITS - starts)).long() + 1
        powers = torch.tensor([10 ** i for i in range(T)], dtype=torch.long, device=device)
        chain_values = (powers[lengths] - 1) * powers[starts]
        trigger = powers[starts]
        mode = idx.remainder(4)
        av = torch.where(mode == 0, chain_values, torch.where(mode == 1, chain_values, torch.where(mode == 2, 99_999_999_999_999 - chain_values, chain_values)))
        bv = torch.where(mode == 0, trigger, torch.where(mode == 1, trigger * 9, torch.where(mode == 2, chain_values, 99_999_999_999_999 - chain_values)))
        for p in range(T):
            a[count - special:count, p] = av.remainder(10)
            b[count - special:count, p] = bv.remainder(10)
            av.div_(10, rounding_mode="floor")
            bv.div_(10, rounding_mode="floor")
    return a, b, labels_for(a, b)


def integer_digits(values):
    x = values.clone()
    out = torch.zeros((len(values), T), dtype=torch.long)
    for p in range(T):
        out[:, p] = x.remainder(10)
        x.div_(10, rounding_mode="floor")
    return out


def random_validation(n, seed=918273):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randint(0, 100_000_000_000_000, (n,), generator=generator)
    b = torch.randint(0, 100_000_000_000_000, (n,), generator=generator)
    ad, bd = integer_digits(a), integer_digits(b)
    return ad, bd, labels_for(ad, bd)


def edge_validation():
    m = 99_999_999_999_999
    pairs = [(0, 0), (m, 0), (m, m), (m, 1), (1, m)]
    for start in range(14):
        for length in range(1, 15 - start):
            prefix = 10 ** start
            chain = (10 ** length - 1) * prefix
            pairs.extend([(chain, prefix), (chain, 9 * prefix), (chain, m - chain)])
    for digit in range(10):
        repeated = int(str(digit) * 14)
        pairs.extend([(repeated, repeated), (repeated, m - repeated)])
    a = torch.tensor([x for x, _ in pairs])
    b = torch.tensor([y for _, y in pairs])
    ad, bd = integer_digits(a), integer_digits(b)
    return ad, bd, labels_for(ad, bd), pairs


@torch.no_grad()
def exact_accuracy(model, data, batch=4096):
    a, b, y = data[:3]
    correct = 0
    for i in range(0, len(a), batch):
        pred = model(a[i:i + batch], b[i:i + batch]).argmax(-1)
        correct += (pred == y[i:i + batch]).all(1).sum().item()
    return correct / len(a)


def export(model, path):
    source = Path(path).read_text()
    marker = "_WEIGHTS = None"
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().tolist()
    literal = "_WEIGHTS = " + repr(weights)
    if marker in source:
        source = source.replace(marker, literal)
    else:
        start = source.index("_WEIGHTS = [")
        end = source.index("\n\n\ndef build_model", start)
        source = source[:start] + literal + source[end:]
    Path(path).write_text(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=71237)
    parser.add_argument("--output", default="/workspace/submission.py")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(13)
    model = AdderTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003)
    random_val = random_validation(20000)
    edges = edge_validation()
    best = None
    best_score = (-1, -1)
    for step in range(1, args.steps + 1):
        model.train()
        fraction = 0.35 if step < args.steps * 0.7 else 0.55
        a, b, y = make_batch(args.batch, fraction, "cpu")
        logits = model(a, b)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == int(args.steps * 0.55):
            for group in optimizer.param_groups: group["lr"] = 1e-3
        if step == int(args.steps * 0.8):
            for group in optimizer.param_groups: group["lr"] = 3e-4
        if step % 250 == 0 or step == args.steps:
            model.eval()
            random_acc = exact_accuracy(model, random_val)
            edge_acc = exact_accuracy(model, edges)
            score = (edge_acc, random_acc)
            if score > best_score:
                best_score = score
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(best, "/workspace/best.pt")
                export(model, args.output)
            print(f"step={step} loss={loss.item():.5f} random={random_acc:.5%} edge={edge_acc:.5%} best={best_score}", flush=True)
    if best is not None:
        model.load_state_dict(best)
        export(model, args.output)
    print("parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
