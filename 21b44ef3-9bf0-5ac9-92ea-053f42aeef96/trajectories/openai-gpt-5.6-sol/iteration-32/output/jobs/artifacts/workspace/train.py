import math
import os
import time
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = "cuda"
BATCH = 8192
LOW, HIGH = 10_000_000, 100_000_000
POW10 = torch.tensor([10, 100, 1000, 10000, 100000, 1000000, 10000000], device=DEVICE, dtype=torch.long)


class TrainAdder(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        self.ff = ff
        self.token = nn.Parameter(torch.empty(11, 20))
        self.pos_coeff = nn.Parameter(torch.empty(25, 2))
        self.pos_basis = nn.Parameter(torch.empty(2, 20))
        self.q = nn.Parameter(torch.empty(2, 20, 20))
        self.o = nn.Parameter(torch.empty(2, 20, 20))
        self.k = nn.Parameter(torch.empty(5, 20))
        self.v = nn.Parameter(torch.empty(5, 20))
        self.norm_weight = nn.Parameter(torch.ones(20))
        self.norm_bias = nn.Parameter(torch.zeros(20))
        self.ff1_weight = nn.Parameter(torch.empty(ff, 20))
        self.ff1_bias = nn.Parameter(torch.zeros(ff))
        self.ff2_weight = nn.Parameter(torch.empty(20, ff))
        self.head_weight = nn.Parameter(torch.empty(10, 20))
        self.head_bias = nn.Parameter(torch.zeros(10))
        self.register_buffer("causal", torch.tril(torch.ones(25, 25, dtype=torch.bool)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token, std=.3)
        nn.init.normal_(self.pos_coeff, std=.3)
        nn.init.normal_(self.pos_basis, std=.3)
        for p in (self.q, self.o, self.k, self.v, self.ff1_weight, self.ff2_weight, self.head_weight):
            nn.init.xavier_uniform_(p.view(-1, p.shape[-1]))

    def forward(self, tokens):
        n = tokens.shape[1]
        x = F.embedding(tokens, self.token) + (self.pos_coeff @ self.pos_basis)[:n]
        mask = self.causal[:n, :n]
        for layer in range(2):
            z = F.layer_norm(x, (20,), self.norm_weight, self.norm_bias)
            q = F.linear(z, self.q[layer]).view(-1, n, 4, 5).transpose(1, 2)
            k, v = F.linear(z, self.k), F.linear(z, self.v)
            score = torch.einsum("bhtd,bsd->bhts", q, k) * (5 ** -.5)
            attn = score.masked_fill(~mask, -torch.inf).softmax(-1)
            mix = torch.einsum("bhts,bsd->bhtd", attn, v).transpose(1, 2).reshape(-1, n, 20)
            x = x + F.linear(mix, self.o[layer])
            z = F.layer_norm(x, (20,))
            x = x + F.linear(F.gelu(F.linear(z, self.ff1_weight, self.ff1_bias)), self.ff2_weight)
        return F.linear(F.layer_norm(x, (20,)), self.head_weight, self.head_bias)


def uniform(n):
    return (torch.randint(LOW, HIGH, (n,), device=DEVICE),
            torch.randint(LOW, HIGH, (n,), device=DEVICE))


def structured(n):
    a, b = uniform(n)
    mode = torch.randint(0, 8, (n,), device=DEVICE)
    # Complements near 100,000,000 and 110,000,000, including exact cases.
    m = mode == 0
    delta = torch.randint(-20, 21, (n,), device=DEVICE)
    bb = 100_000_000 - a + delta
    ok = (bb >= LOW) & (bb < HIGH)
    b = torch.where(m & ok, bb, b)
    m = mode == 1
    bb = 110_000_000 - a + delta
    ok = (bb >= LOW) & (bb < HIGH)
    b = torch.where(m & ok, bb, b)
    # Force arbitrary-length lower suffix sums around a carry boundary.
    m = mode == 2
    p = POW10[torch.randint(0, 7, (n,), device=DEVICE)]
    al = a.remainder(p)
    d = torch.randint(-10, 11, (n,), device=DEVICE)
    bl = torch.minimum(torch.maximum(p - al + d, torch.zeros_like(p)), p - 1)
    b2 = torch.div(b, p, rounding_mode="floor") * p + bl
    b = torch.where(m & (b2 >= LOW) & (b2 < HIGH), b2, b)
    # Long asymmetric 0/9 suffixes.
    m = mode == 3
    p = POW10[torch.randint(0, 7, (n,), device=DEVICE)]
    side = torch.randint(0, 2, (n,), device=DEVICE).bool()
    a2 = torch.div(a, p, rounding_mode="floor") * p + torch.where(side, p - 1, torch.zeros_like(p))
    b2 = torch.div(b, p, rounding_mode="floor") * p + torch.where(side, torch.ones_like(p), p - 1)
    a, b = torch.where(m, a2, a), torch.where(m, b2, b)
    # Decimal-rounded operands with small asymmetric offsets.
    m = mode == 4
    p = POW10[torch.randint(1, 7, (n,), device=DEVICE)]
    offs = torch.randint(-9, 10, (n,), device=DEVICE)
    a2 = (torch.div(a, p, rounding_mode="floor") * p + offs).clamp(LOW, HIGH - 1)
    b2 = (torch.div(b, p, rounding_mode="floor") * p - offs).clamp(LOW, HIGH - 1)
    a, b = torch.where(m, a2, a), torch.where(m, b2, b)
    # Repeated-digit and extreme operands.
    m = mode == 5
    reps = torch.tensor([11_111_111,22_222_222,33_333_333,44_444_444,55_555_555,66_666_666,77_777_777,88_888_888,99_999_999], device=DEVICE)
    a2 = reps[torch.randint(0, 9, (n,), device=DEVICE)]
    b2 = reps[torch.randint(0, 9, (n,), device=DEVICE)]
    a, b = torch.where(m, a2, a), torch.where(m, b2, b)
    m = mode == 6
    edge = torch.tensor([10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_000_000,19_999_999,20_000_000,89_999_999,90_000_000,98_999_999,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_999], device=DEVICE)
    a2 = edge[torch.randint(0, len(edge), (n,), device=DEVICE)]
    b2 = edge[torch.randint(0, len(edge), (n,), device=DEVICE)]
    a, b = torch.where(m, a2, a), torch.where(m, b2, b)
    # Sparse internal nonzero digits.
    m = mode == 7
    place = (10 ** torch.randint(0, 7, (n,), device=DEVICE))
    a2 = 10_000_000 + torch.randint(0, 9, (n,), device=DEVICE) * place
    b2 = 90_000_000 + torch.randint(0, 10, (n,), device=DEVICE) * place
    b2 = b2.clamp(max=HIGH - 1)
    a, b = torch.where(m, a2, a), torch.where(m, b2, b)
    return a, b


def batch(n=BATCH, structured_fraction=.4):
    a, b = uniform(n)
    count = int(n * structured_fraction)
    if count:
        sa, sb = structured(count)
        a[:count], b[:count] = sa, sb
    return encode(a, b)


def digits(x, count):
    out = []
    for _ in range(count):
        out.append(x.remainder(10))
        x = torch.div(x, 10, rounding_mode="floor")
    return torch.stack(out, 1)


def encode(a, b):
    da, db = digits(a, 8), digits(b, 8)
    operands = torch.stack((da, db), 2).reshape(-1, 16)
    target = digits(a + b, 9)
    inp = torch.cat((operands, torch.full((a.numel(), 1), 10, device=DEVICE), target[:, :8]), 1)
    return inp, target


@torch.no_grad()
def evaluate(model, n=100000, kind="uniform", chunk=10000):
    model.eval()
    good = total = 0
    min_margin = 1e9
    while total < n:
        size = min(chunk, n - total)
        a, b = uniform(size) if kind == "uniform" else structured(size)
        da, db = digits(a, 8), digits(b, 8)
        seq = torch.stack((da, db), 2).reshape(-1, 16)
        seq = torch.cat((seq, torch.full((size, 1), 10, device=DEVICE)), 1)
        pred = []
        margins = []
        for _ in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, 1).values
            margins.append(top[:, 0] - top[:, 1])
            d = logits.argmax(1)
            pred.append(d)
            seq = torch.cat((seq, d[:, None]), 1)
        pred = torch.stack(pred, 1)
        target = digits(a + b, 9)
        good += (pred == target).all(1).sum().item()
        min_margin = min(min_margin, torch.stack(margins, 1).min().item())
        total += size
    model.train()
    return good / total, min_margin


def save(model, name):
    path = f"/workspace/{name}.pt"
    temp = path + ".new"
    torch.save({"ff": model.ff, "state": model.state_dict()}, temp)
    os.replace(temp, path)


def train_phase(model, steps, lr, structured_fraction, name, warmup=0, validate_every=2000):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=.01)
    started = time.time()
    for step in range(1, steps + 1):
        if warmup:
            scale = min(1., step / warmup)
            # cosine decay to 15% after warmup
            if step > warmup:
                scale = .15 + .85 * .5 * (1 + math.cos(math.pi * (step-warmup)/(steps-warmup)))
            opt.param_groups[0]["lr"] = lr * scale
        x, y = batch(structured_fraction=structured_fraction)
        logits = model(x)[:, 16:]
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        opt.step()
        if step == 1 or step % 200 == 0:
            print(f"{name} {step}/{steps} loss={loss.item():.6f} lr={opt.param_groups[0]['lr']:.2g} time={time.time()-started:.1f}", flush=True)
        if step % validate_every == 0 or step == steps:
            ua, um = evaluate(model, 20000, "uniform")
            sa, sm = evaluate(model, 20000, "structured")
            print(f"VALID {name} step={step} uniform={ua:.6f} structured={sa:.6f} margins={um:.3f}/{sm:.3f}", flush=True)
            save(model, name)
    return model


def prune(model, new_ff):
    assert new_ff == model.ff - 1
    importance = model.ff1_weight.norm(dim=1) * model.ff2_weight.norm(dim=0)
    keep = torch.argsort(importance, descending=True)[:new_ff].sort().values
    out = TrainAdder(new_ff).to(DEVICE)
    old = model.state_dict()
    state = out.state_dict()
    for key in state:
        if key == "ff1_weight": state[key].copy_(old[key][keep])
        elif key == "ff1_bias": state[key].copy_(old[key][keep])
        elif key == "ff2_weight": state[key].copy_(old[key][:, keep])
        else: state[key].copy_(old[key])
    out.load_state_dict(state)
    return out


def main():
    torch.manual_seed(320031)
    torch.set_float32_matmul_precision("high")
    model = TrainAdder(4).to(DEVICE)
    train_phase(model, 36000, 2e-3, .35, "teacher", warmup=1000, validate_every=3000)
    train_phase(model, 6000, 6e-5, .5, "teacher_stable", validate_every=2000)
    model = prune(model, 3)
    train_phase(model, 18000, 2e-5, .5, "width3", validate_every=3000)
    model = prune(model, 2)
    train_phase(model, 30000, 1e-5, .5, "width2a", validate_every=3000)
    train_phase(model, 16000, 3e-6, .55, "width2_final", validate_every=2000)
    save(model, "final_full")
    print("FINAL", evaluate(model, 500000, "uniform"), evaluate(model, 500000, "structured"), flush=True)


if __name__ == "__main__":
    main()
