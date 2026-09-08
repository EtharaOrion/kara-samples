import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class AdditionTransformer(nn.Module):
    def __init__(self, ff_width=4):
        super().__init__()
        d = 20
        self.token = nn.Embedding(11, d)
        self.pos_left = nn.Parameter(torch.empty(25, 2))
        self.pos_right = nn.Parameter(torch.empty(2, d))
        self.q = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.k = nn.Linear(d, 5, bias=False)
        self.v = nn.Linear(d, 5, bias=False)
        self.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(2)])
        self.attn_norm = nn.LayerNorm(d)
        self.ff_norm = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, ff_width, bias=False)
        self.ff_out = nn.Linear(ff_width, d, bias=False)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 10, bias=False)
        self.register_buffer("causal", torch.triu(torch.ones(25, 25, dtype=torch.bool), 1), persistent=False)
        nn.init.normal_(self.pos_left, std=0.12)
        nn.init.normal_(self.pos_right, std=0.12)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.pos_left[:length] @ self.pos_right
        mask = self.causal[:length, :length]
        for layer in range(2):
            z = self.attn_norm(x)
            q = self.q[layer](z).view(tokens.shape[0], length, 4, 5).transpose(1, 2)
            k, v = self.k(z), self.v(z)
            scores = torch.einsum("bhtd,bsd->bhts", q, k) * 0.4472135954999579
            attention = scores.masked_fill(mask, -torch.inf).softmax(-1)
            context = torch.einsum("bhts,bsd->bhtd", attention, v)
            x = x + self.o[layer](context.transpose(1, 2).reshape(tokens.shape[0], length, 20))
            x = x + self.ff_out(F.gelu(self.ff_in(self.ff_norm(x))))
        return self.output(self.final_norm(x))


DEVICE = "cuda"
LOW = 10_000_000
HIGH = 99_999_999
POW10 = torch.tensor([10 ** i for i in range(9)], device=DEVICE, dtype=torch.long)


def make_batch(n, structured=0.18):
    a = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
    m = int(n * structured)
    if m:
        x = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        y = torch.randint(LOW, HIGH + 1, (m,), device=DEVICE)
        kind = torch.randint(0, 8, (m,), device=DEVICE)
        k = torch.randint(1, 8, (m,), device=DEVICE)
        scale = 10 ** k
        # Exact and near complements exercise long carries at every position.
        comp = 100_000_000 - x + torch.randint(-2, 3, (m,), device=DEVICE)
        y = torch.where(kind <= 1, comp, y)
        # Asymmetric runs of trailing nines and zeros.
        nines = (x // scale) * scale + scale - 1
        zeros = (x // scale) * scale
        y = torch.where(kind == 2, nines, y)
        y = torch.where(kind == 3, zeros, y)
        # Decimal boundaries with independent random counterpart.
        boundary = (torch.randint(1, 100, (m,), device=DEVICE) * scale + torch.randint(-2, 3, (m,), device=DEVICE))
        y = torch.where(kind == 4, boundary, y)
        # Repeated and sparse digit patterns.
        digit = torch.randint(0, 10, (m,), device=DEVICE)
        repeated = digit * 11_111_111
        y = torch.where(kind == 5, repeated, y)
        sparse = torch.randint(1, 10, (m,), device=DEVICE) * (10 ** torch.randint(6, 8, (m,), device=DEVICE))
        sparse += torch.randint(0, 10, (m,), device=DEVICE) * (10 ** torch.randint(0, 6, (m,), device=DEVICE))
        y = torch.where(kind == 6, sparse, y)
        # Near extrema.
        edge = torch.where(torch.rand(m, device=DEVICE) < .5,
                           LOW + torch.randint(0, 10001, (m,), device=DEVICE),
                           HIGH - torch.randint(0, 10001, (m,), device=DEVICE))
        y = torch.where(kind == 7, edge, y)
        a[:m] = x.clamp(LOW, HIGH)
        b[:m] = y.clamp(LOW, HIGH)
    return a, b


def encode(a, b):
    da = (a[:, None] // POW10[:8]) % 10
    db = (b[:, None] // POW10[:8]) % 10
    prefix = torch.empty(a.shape[0], 17, dtype=torch.long, device=DEVICE)
    prefix[:, 0:16:2] = da
    prefix[:, 1:16:2] = db
    prefix[:, 16] = 10
    target = ((a + b)[:, None] // POW10) % 10
    tokens = torch.cat((prefix, target[:, :8]), dim=1)
    return tokens, target


@torch.no_grad()
def evaluate(model, n=20000, structured=0.0, batch=5000):
    model.eval()
    good = total = 0
    min_margin = 1e9
    for _ in range((n + batch - 1) // batch):
        size = min(batch, n - total)
        a, b = make_batch(size, structured)
        seq, target = encode(a, b)
        seq = seq[:, :17]
        pred = []
        for j in range(9):
            logits = model(seq)[:, -1]
            top = logits.topk(2, dim=-1).values
            min_margin = min(min_margin, float((top[:, 0] - top[:, 1]).min()))
            d = logits.argmax(-1)
            pred.append(d)
            if j != 8:
                seq = torch.cat((seq, d[:, None]), 1)
        pred = torch.stack(pred, 1)
        good += int((pred == target).all(1).sum())
        total += size
    model.train()
    return good, total, min_margin


def prune(model, new_width):
    old = model.ff_in.out_features
    assert new_width == old - 1
    score = model.ff_in.weight.detach().norm(dim=1) * model.ff_out.weight.detach().norm(dim=0)
    keep = [i for i in range(old) if i != int(score.argmin())]
    result = AdditionTransformer(new_width).to(DEVICE)
    state = model.state_dict()
    fresh = result.state_dict()
    for name in fresh:
        if name == "ff_in.weight":
            fresh[name].copy_(state[name][keep])
        elif name == "ff_out.weight":
            fresh[name].copy_(state[name][:, keep])
        else:
            fresh[name].copy_(state[name])
    result.load_state_dict(fresh)
    return result


def train_phase(model, steps, lr, structured, label, save_every=2000):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, .98), weight_decay=0.01, fused=True)
    start = time.time()
    for step in range(1, steps + 1):
        a, b = make_batch(8192, structured)
        tokens, target = encode(a, b)
        logits = model(tokens)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % save_every == 0 or step == steps:
            torch.save(model.state_dict(), f"/workspace/{label}.pt")
            g, n, margin = evaluate(model, 20000, 0.0)
            gs, ns, _ = evaluate(model, 20000, 0.65)
            print(f"{label} {step}/{steps} loss={loss.item():.5f} random={g}/{n} structured={gs}/{ns} margin={margin:.3f} elapsed={time.time()-start:.1f}", flush=True)
    return model


def export(model):
    source_path = Path("/workspace/submission.py")
    source = source_path.read_text()
    start = source.index("_TRAINED_STATE = ")
    end = source.index("\n\n\ndef build_model", start)
    values = []
    for name, tensor in model.state_dict().items():
        flat = tensor.detach().cpu().float().reshape(-1).tolist()
        literal = ",".join(format(x, ".9g") for x in flat)
        values.append(f"    {name!r}: torch.tensor([{literal}]).reshape{tuple(tensor.shape)},")
    block = "_TRAINED_STATE = {\n" + "\n".join(values) + "\n}"
    source = source[:start] + block + source[end:]
    source = source.replace("model = AdditionTransformer(4)", f"model = AdditionTransformer({model.ff_in.out_features})")
    source_path.write_text(source)
    print("exported", sum(p.numel() for p in model.parameters()), "parameters to", source_path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2025)
    random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer(4).to(DEVICE)
    if args.quick:
        model = train_phase(model, 8000, 2e-3, .18, "teacher", 1000)
        export(model)
        return
    model = train_phase(model, 36000, 2e-3, .18, "teacher", 2000)
    model = train_phase(model, 6000, 8e-5, .30, "teacher_stable", 2000)
    model = prune(model, 3)
    model = train_phase(model, 18000, 2e-5, .30, "width3", 2000)
    model = prune(model, 2)
    model = train_phase(model, 30000, 1.2e-5, .38, "width2", 2000)
    model = train_phase(model, 12000, 4e-6, .40, "width2_polish", 2000)
    export(model)


if __name__ == "__main__":
    main()
