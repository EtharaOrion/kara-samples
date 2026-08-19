import random
import torch
from submission import AdditionTransformer
from train import DEVICE, LIMIT, POW10, columns, evaluate


def edge_pairs():
    pairs = {(0, 0), (LIMIT - 1, 0), (LIMIT - 1, 1), (LIMIT - 1, LIMIT - 1)}
    for start in range(14):
        p = 10 ** start
        for length in range(1, 15 - start):
            run = (10 ** length - 1) * p
            for increment in range(1, 10):
                if run + increment * p < LIMIT:
                    pairs.add((run, increment * p))
                    pairs.add((increment * p, run))
                near = run - p
                pairs.add((near, increment * p))
                pairs.add((increment * p, near))
    rep = (10 ** 14 - 1) // 9
    for x in range(10):
        for y in range(10):
            pairs.add((x * rep, y * rep))
    for exponent in range(15):
        boundary = 10 ** exponent
        for delta in range(1, 20):
            if 0 <= boundary - delta < LIMIT:
                pairs.add((boundary - delta, delta))
                pairs.add((delta, boundary - delta))
    return sorted(pairs)


@torch.inference_mode()
def decode(model, a, b):
    count = len(a)
    a = torch.tensor(a, device=DEVICE)
    b = torch.tensor(b, device=DEVICE)
    target = columns(a + b, 15)
    ta = torch.cat((columns(a), torch.full((count, 1), 10, device=DEVICE)), 1)
    tb = torch.cat((columns(b), torch.full((count, 1), 10, device=DEVICE)), 1)
    out = torch.full((count, 15), 10, device=DEVICE)
    result = []
    for _ in range(15):
        digit = model(ta, tb, out)[:, -1].argmax(1)
        result.append(digit)
        ta = torch.cat((ta, torch.full((count, 1), 10, device=DEVICE)), 1)
        tb = torch.cat((tb, torch.full((count, 1), 10, device=DEVICE)), 1)
        out = torch.cat((out, digit[:, None]), 1)
    result = torch.stack(result, 1)
    wrong = (result != target).any(1)
    return int(wrong.sum())


def attention_change(model):
    block = model.blocks[0]
    a1 = torch.tensor([[1, 2, 3, 4]], device=DEVICE)
    a2 = torch.tensor([[9, 8, 7, 6]], device=DEVICE)
    b = torch.tensor([[2, 3, 4, 5]], device=DEVICE)
    o = torch.full_like(a1, 10)
    pos = torch.arange(4, device=DEVICE)
    def scores(a):
        x = model.operand_a(a) + model.operand_b(b) + model.output(o) + model.position(pos)
        qkv = block.qkv(block.norm1(x)).reshape(1, 4, 3, block.heads, -1)
        q, k, _ = qkv.unbind(2)
        return torch.einsum("bthd,bshd->bhts", q, k)
    return float((scores(a1) - scores(a2)).abs().max())


model = AdditionTransformer().to(DEVICE)
model.load_state_dict(torch.load('/workspace/model.pt', weights_only=True))
model.eval()
print('parameters', sum(p.numel() for p in model.parameters()))
# uniform done
# structured done
pairs = edge_pairs()
print('edges', decode(model, [a for a, _ in pairs], [b for _, b in pairs]), '/', len(pairs))
print('attention_change', attention_change(model))
original = model.head.weight.detach().clone()
with torch.no_grad(): model.head.weight.zero_()
random.seed(7)
sample = [(random.randrange(LIMIT), random.randrange(LIMIT)) for _ in range(4096)]
print('zero_head_errors', decode(model, [a for a, _ in sample], [b for _, b in sample]), '/ 4096')
with torch.no_grad(): model.head.weight.copy_(original)
