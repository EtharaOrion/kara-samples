import sys
import torch
sys.path.insert(0, "/workspace")
from submission import build_model, add
from train import batch_data, digits, patterned_digits, values


def evaluate(model, batches, maker):
    errors = total = 0
    examples = []
    for _ in range(batches):
        aa, bb, target = maker()
        prefix = torch.full((aa.shape[0], 1), 10, dtype=torch.long, device=aa.device)
        outputs = []
        for _ in range(15):
            next_digit = model(aa, bb, prefix)[:, -1].argmax(-1)
            outputs.append(next_digit)
            prefix = torch.cat((prefix, next_digit[:, None]), 1)
        pred = torch.stack(outputs, 1)
        bad = (pred != target).any(1)
        errors += int(bad.sum())
        total += aa.shape[0]
        if bad.any() and len(examples) < 5:
            rows = torch.where(bad)[0][:5-len(examples)]
            examples.extend(zip(values(aa[rows, :14]).tolist(), values(bb[rows, :14]).tolist(), pred[rows].tolist(), target[rows].tolist()))
    return errors, total, examples


def packed(ad, bd):
    device = ad.device
    n = ad.shape[0]
    pad = torch.full((n, 1), 10, dtype=torch.long, device=device)
    aa = torch.cat((ad, pad), 1)
    bb = torch.cat((bd, pad), 1)
    target = digits(values(ad) + values(bd), 15)
    return aa, bb, target


def systematic(device):
    pairs = []
    for p in range(14):
        for x in range(10):
            for y in range(10):
                a = [0] * 14; b = [0] * 14
                a[p] = x; b[p] = y
                pairs.append((a, b))
        for length in range(1, 15-p):
            a = [0] * 14; b = [0] * 14
            for q in range(p, p + length): a[q] = 9
            b[p] = 1
            pairs.append((a, b))
            a2 = [0] * 14; b2 = [0] * 14
            for q in range(p, p + length): a2[q] = 8
            b2[p] = 1
            pairs.append((a2, b2))
    ad = torch.tensor([x[0] for x in pairs], device=device)
    bd = torch.tensor([x[1] for x in pairs], device=device)
    return packed(ad, bd)


def main():
    torch.manual_seed(8821)
    device = torch.device("cuda")
    model, metadata = build_model()
    model.to(device).eval()
    print("parameters", sum(p.numel() for p in model.parameters()), metadata)
    def random_maker():
        aa, bb, _, target = batch_data(4096, device, 0.0)
        return aa, bb, target
    def mixed_maker():
        aa, bb, _, target = batch_data(4096, device, 0.55, 0.20)
        return aa, bb, target
    print("uniform", evaluate(model, 128, random_maker))
    print("mixed", evaluate(model, 128, mixed_maker))
    for kind in range(7):
        def maker(kind=kind):
            ad, bd = patterned_digits(4096, device, kind)
            return packed(ad, bd)
        print("kind", kind, evaluate(model, 32, maker))
    fixed = systematic(device)
    print("systematic", evaluate(model, 1, lambda: fixed))
    block = model.blocks[0]
    x1 = model.a_embed(fixed[0][:2]) + model.b_embed(fixed[1][:2]) + model.position[:15]
    z = block.ln1(x1)
    q, k, _ = block.qkv(z).chunk(3, -1)
    q = q.view(2, 15, 3, 3).transpose(1, 2)
    k = k.view(2, 15, 3, 3).transpose(1, 2)
    scores = q @ k.transpose(-2, -1)
    print("qk input difference", float((scores[0] - scores[1]).abs().max()))
    cpu_model = model.cpu()
    for a, b in [(0, 0), (2, 3), (99_999_999_999_999, 1), (9_000_000_000_000, 9_000_000_000_000), (99_999_999_999_999, 99_999_999_999_999)]:
        print("add", a, b, add(cpu_model, a, b))

if __name__ == "__main__": main()
