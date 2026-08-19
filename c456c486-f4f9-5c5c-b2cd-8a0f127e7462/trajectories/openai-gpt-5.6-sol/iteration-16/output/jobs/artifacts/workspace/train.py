import copy
import random
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda"
MAXIMUM = 99_999_999_999_999
WIDTH = 16
HEADS = 2
HIDDEN = 32
LAYERS = 3
BATCH = 4096


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.fc1 = nn.Linear(WIDTH, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, WIDTH)

    def forward(self, x, mask):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        size = width // HEADS
        q = q.view(batch, length, HEADS, size).transpose(1, 2)
        k = k.view(batch, length, HEADS, size).transpose(1, 2)
        v = v.view(batch, length, HEADS, size).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (size ** -0.5)
        scores.masked_fill_(mask, -1e4)
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(13, WIDTH)
        self.position = nn.Parameter(torch.empty(45, WIDTH))
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, 10)
        self.register_buffer("causal", torch.triu(torch.ones(45, 45, dtype=torch.bool), 1), persistent=False)
        nn.init.normal_(self.position, std=.02)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position[:length]
        for block in self.blocks:
            x = block(x, self.causal[:length, :length])
        return self.output(self.norm(x))


POWERS = (10 ** torch.arange(14, device=DEVICE, dtype=torch.long))
OUT_POWERS = (10 ** torch.arange(15, device=DEVICE, dtype=torch.long))


def digits(values, powers):
    return (values[:, None] // powers[None]) % 10


def random_batch(batch, structured=.2):
    a = torch.randint(0, MAXIMUM + 1, (batch,), device=DEVICE)
    b = torch.randint(0, MAXIMUM + 1, (batch,), device=DEVICE)
    count = int(batch * structured)
    if count:
        # Randomized full-number carry chains: low k digits sum to 9, then an increment propagates.
        k = torch.randint(1, 15, (count,), device=DEVICE)
        p = 10 ** k
        low = torch.randint(0, MAXIMUM + 1, (count,), device=DEVICE) % p
        a[:count] = torch.randint(0, MAXIMUM + 1, (count,), device=DEVICE)
        a[:count] = (a[:count] // p) * p + low
        b[:count] = p - low
        b[:count].clamp_(max=MAXIMUM)
        swap = torch.rand(count, device=DEVICE) < .5
        left = a[:count].clone()
        a[:count] = torch.where(swap, b[:count], a[:count])
        b[:count] = torch.where(swap, left, b[:count])
    return a, b


def encode(a, b, include_output=True):
    batch = a.shape[0]
    source = torch.empty((batch, 45 if include_output else 30), dtype=torch.long, device=DEVICE)
    source[:, 0] = 10
    source[:, 1:29:2] = digits(a, POWERS)
    source[:, 2:29:2] = digits(b, POWERS)
    source[:, 29] = 12
    if include_output:
        source[:, 30:45] = digits(a + b, OUT_POWERS)
    return source


@torch.no_grad()
def evaluate(model, count=32768, structured=0.0):
    model.eval()
    a, b = random_batch(count, structured)
    sequence = encode(a, b, False)
    for _ in range(15):
        prediction = model(sequence)[:, -1].argmax(-1, keepdim=True)
        sequence = torch.cat((sequence, prediction), 1)
    expected = digits(a + b, OUT_POWERS)
    correct = (sequence[:, 30:] == expected).all(1)
    model.train()
    return correct.float().mean().item(), a[~correct][:10].cpu(), b[~correct][:10].cpu()


@torch.no_grad()
def edge_evaluate(model):
    cases = [(0, 0), (MAXIMUM, 0), (MAXIMUM, 1), (MAXIMUM, MAXIMUM)]
    for k in range(1, 15):
        p = 10 ** k
        cases.extend([(p - 1, 1), (p - 1, p - 1), (MAXIMUM - (p - 1), p - 1)])
        for d in (1, 3, 7, 9):
            cases.append((d * (p - 1) // 9, p))
    a = torch.tensor([x for x, _ in cases], device=DEVICE)
    b = torch.tensor([y for _, y in cases], device=DEVICE)
    sequence = encode(a, b, False)
    for _ in range(15):
        sequence = torch.cat((sequence, model(sequence)[:, -1].argmax(-1, keepdim=True)), 1)
    return (sequence[:, 30:] == digits(a + b, OUT_POWERS)).all(1).float().mean().item()


def main():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1601)
    model = Model().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.005)
    best = 0.0
    for step in range(1, 20001):
        if step == 7001:
            for group in optimizer.param_groups: group["lr"] = 1e-3
        if step == 13001:
            for group in optimizer.param_groups: group["lr"] = 3e-4
        if step == 17001:
            for group in optimizer.param_groups: group["lr"] = 1e-4
        a, b = random_batch(BATCH, .15 if step < 13000 else .3)
        sequence = encode(a, b)
        logits = model(sequence[:, :-1])[:, 29:44]
        loss = F.cross_entropy(logits.reshape(-1, 10), sequence[:, 30:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            accuracy, bad_a, bad_b = evaluate(model, 32768, .0)
            edge = edge_evaluate(model)
            print(step, f"loss={loss.item():.5g} random={accuracy:.6f} edge={edge:.4f}", flush=True)
            if accuracy + edge > best:
                best = accuracy + edge
                torch.save(model.state_dict(), "/workspace/model.pt")
            if accuracy >= .99995 and edge == 1.0 and step >= 10000:
                torch.save(model.state_dict(), "/workspace/model.pt")
                break
    model.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
    print("final random", evaluate(model, 262144, 0.0)[0], "structured", evaluate(model, 131072, .8)[0], "edge", edge_evaluate(model), flush=True)


if __name__ == "__main__":
    main()
