import torch
from submission import AdditionTransformer
from train import make_batch, evaluate, targets, DEVICE


def from_int(values):
    values = torch.tensor(values, dtype=torch.long, device=DEVICE)
    digits = []
    for _ in range(14):
        digits.append(values.remainder(10))
        values = values.div(10, rounding_mode="floor")
    return torch.stack(digits, 1)


@torch.no_grad()
def check_pairs(model, pairs, batch_size=16384):
    failures = []
    for offset in range(0, len(pairs), batch_size):
        part = pairs[offset:offset + batch_size]
        a = from_int([x for x, _ in part])
        b = from_int([y for _, y in part])
        expected = targets(a, b)
        pad = torch.full((len(part), 1), 10, dtype=torch.long, device=DEVICE)
        aa = torch.cat((a, pad), 1)
        bb = torch.cat((b, pad), 1)
        previous = pad
        output = []
        for _ in range(15):
            digit = model(aa, bb, previous)[:, -1].argmax(-1, keepdim=True)
            output.append(digit)
            previous = torch.cat((previous, digit), 1)
        bad = (torch.cat(output, 1) != expected).any(1).cpu().tolist()
        failures.extend(part[i] for i, flag in enumerate(bad) if flag)
    return failures


def systematic_pairs():
    limit = 10 ** 14 - 1
    pairs = [(0, 0), (limit, 0), (limit, limit), (limit, 1), (1, limit)]
    values = [0, 1, 4, 5, 9, 10, 11, 49, 50, 99, 100, 101, 499, 500, 999]
    for power in range(14):
        p = 10 ** power
        for x in values:
            if x * p <= limit:
                pairs.extend([(x * p, p), (p, x * p), (x * p, 5 * p), (5 * p, x * p)])
        nines = p - 1
        pairs.extend([(nines, 1), (1, nines), (nines, p), (p, nines)])
        for start in range(power + 1):
            run = (10 ** (power - start + 1) - 1) * (10 ** start)
            if run <= limit:
                pairs.extend([(run, 10 ** start), (10 ** start, run)])
    for length in range(1, 15):
        run = 10 ** length - 1
        pairs.extend([(run, 1), (1, run), (run, run), (run, 5), (5, run)])
        for prefix in (1, 5, 8, 9):
            x = prefix * 10 ** length + run
            if x <= limit:
                pairs.extend([(x, 1), (1, x), (x, 10 ** length)])
    for da in range(10):
        for db in range(10):
            a = int(str(da) * 14)
            b = int(str(db) * 14)
            pairs.extend([(a, b), (b, a)])
    return list(dict.fromkeys((a, b) for a, b in pairs if 0 <= a <= limit and 0 <= b <= limit))


def main():
    model = AdditionTransformer().to(DEVICE)
    model.load_state_dict(torch.load("/workspace/model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    torch.manual_seed(9030)
    e, n = evaluate(model, 64, 0.0)
    s, sn = evaluate(model, 64, 1.0)
    pairs = systematic_pairs()
    failures = check_pairs(model, pairs)
    print("uniform", e, n, "structured", s, sn, "systematic", len(failures), len(pairs))
    print("failures", failures[:30])

    a = torch.tensor([[1] * 15, [9] * 15], device=DEVICE)
    b = torch.tensor([[2] * 15, [0] * 15], device=DEVICE)
    previous = torch.full((2, 1), 10, dtype=torch.long, device=DEVICE)
    block = model.blocks[0]
    x = torch.cat((model.a_embedding(a) + model.b_embedding(b), model.out_embedding(previous)), 1)
    x = x + model.position[:x.shape[1]]
    z = block.ln1(x)
    q, k, _ = block.qkv(z).chunk(3, -1)
    q = q.view(2, -1, 2, 5).transpose(1, 2)
    k = k.view(2, -1, 2, 5).transpose(1, 2)
    scores = q @ k.transpose(-2, -1)
    print("qk_input_difference", (scores[0] - scores[1]).abs().max().item())


if __name__ == "__main__":
    main()
