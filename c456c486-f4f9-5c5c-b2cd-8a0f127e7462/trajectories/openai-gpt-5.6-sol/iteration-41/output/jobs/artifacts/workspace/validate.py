import sys
sys.path.insert(0, '/workspace')
import torch
import train

DEVICE = torch.device('cuda')


def load(path):
    m = train.AddTransformer().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def compare(models, count, structured, boundary):
    errors = [0] * len(models)
    torch.manual_seed(91027 + int(structured * 100) + int(boundary * 1000))
    for start in range(0, count, 4096):
        n = min(4096, count - start)
        a, b = train.random_batch(n, DEVICE, structured, boundary)
        target = train.output_digits(a + b)
        for i, model in enumerate(models):
            errors[i] += (train.decode(model, a, b) != target).any(1).sum().item()
    return errors


def systematic():
    pairs = {(0, 0), (0, 99_999_999_999_999), (99_999_999_999_999, 0),
             (99_999_999_999_999, 1), (99_999_999_999_999, 99_999_999_999_999)}
    for pos in range(14):
        p = 10 ** pos
        for x, y in [(p, p), (5*p, 5*p), (9*p, 9*p), (9*p, p), (8*p, p)]:
            pairs.add((x, y)); pairs.add((y, x))
        for length in range(1, 15-pos):
            hi = 10 ** (pos + length)
            run9 = hi - p
            for x, y in [(run9, p), (run9-p, p), (run9, 2*p), (run9, 9*p)]:
                if 0 <= x < train.MAX_VALUE and 0 <= y < train.MAX_VALUE:
                    pairs.add((x, y)); pairs.add((y, x))
    ones = 11_111_111_111_111
    for x in range(10):
        for y in range(10):
            pairs.add((x*ones, y*ones))
    data = sorted(pairs)
    return (torch.tensor([x for x, _ in data], device=DEVICE),
            torch.tensor([y for _, y in data], device=DEVICE))


@torch.no_grad()
def qk_delta(model):
    a = torch.tensor([12345678901234, 99999999999999], device=DEVICE)
    b = torch.tensor([31415926535897, 1], device=DEVICE)
    x = model.a_embed(train.digits(a)) + model.b_embed(train.digits(b))
    x = x + torch.nn.functional.pad(model.position[:15], (0, 6))
    z = model.attn1.norm(x)
    q, k, _ = model.attn1.qkv(z).chunk(3, -1)
    scores = q.view(2, 15, 3, 3).transpose(1, 2) @ k.view(2, 15, 3, 3).transpose(1, 2).transpose(-1, -2)
    return (scores[0] - scores[1]).abs().max().item()


def main():
    models = [load('/workspace/model.pt'), load('/workspace/model_refined.pt')]
    for label, s, b in [('uniform', 0.0, 0.0), ('mixed', .45, .10), ('structured', 1.0, .12)]:
        print(label, compare(models, 1048576, s, b), flush=True)
    a, b = systematic()
    target = train.output_digits(a+b)
    print('systematic_count', len(a))
    for i, m in enumerate(models):
        pred = train.decode(m, a, b)
        bad = (pred != target).any(1)
        print('model', i, 'systematic_errors', bad.sum().item(), 'qk_delta', qk_delta(m))
        if bad.any():
            indices = bad.nonzero()[:10, 0]
            for j in indices:
                print(int(a[j]), int(b[j]), pred[j].tolist(), target[j].tolist())
    # Demonstrate learned weights are causally used.
    m = models[1]
    probe_a = torch.tensor([12345678901234, 99999999999999], device=DEVICE)
    probe_b = torch.tensor([31415926535897, 1], device=DEVICE)
    before = train.decode(m, probe_a, probe_b)
    saved = m.head.weight.detach().clone()
    with torch.no_grad(): m.head.weight.zero_()
    after = train.decode(m, probe_a, probe_b)
    with torch.no_grad(): m.head.weight.copy_(saved)
    print('head_ablation_changed', bool((before != after).any()))


if __name__ == '__main__':
    main()
