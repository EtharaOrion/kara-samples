import argparse
import importlib.util
import math
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path("/workspace")
DEVICE = "cuda"
BATCH = 8192
LIMIT = 100_000_000
LOW = 10_000_000
POWERS = torch.tensor([10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000], device=DEVICE)
DIVISORS = torch.tensor([10**i for i in range(9)], device=DEVICE)
EDGE_VALUES = torch.tensor(sorted(set(
    [LOW, LOW + 1, LOW + 9, LOW + 10, LOW + 99, LOW + 999, 11_111_111,
     20_000_000, 40_000_000, 50_000_000, 89_999_999, 90_000_000,
     98_999_999, 99_000_000, 99_899_999, 99_900_000, 99_990_000,
     99_999_000, 99_999_900, 99_999_990, 99_999_998, 99_999_999] +
    [x for p in range(1, 8) for x in (LIMIT // 10**p * 10**p - 1,
                                      LIMIT // 10**p * 10**p - 10**(p-1),
                                      LOW + 10**p - 1, LOW + 10**p)]
)), device=DEVICE)


def load_submission():
    spec = importlib.util.spec_from_file_location("submission_train", ROOT / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digits(x, width=9):
    return torch.remainder(x[:, None] // DIVISORS[:width], 10)


def make_batch(batch=BATCH, structured=0.35):
    a = torch.randint(LOW, LIMIT, (batch,), device=DEVICE)
    b = torch.randint(LOW, LIMIT, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        mode = torch.randint(0, 6, (n,), device=DEVICE)
        # Exact and near complements exercise carries through every column.
        idx = mode == 0
        count = int(idx.sum())
        if count:
            aa = torch.randint(LOW, LIMIT, (count,), device=DEVICE)
            total = LIMIT + torch.randint(-999, 1000, (count,), device=DEVICE)
            bb = total - aa
            valid = (bb >= LOW) & (bb < LIMIT)
            bb = torch.where(valid, bb, LIMIT - aa)
            a[:n][idx], b[:n][idx] = aa, bb
        # Deliberately start/stop a carry at varying suffix lengths.
        idx = mode == 1
        count = int(idx.sum())
        if count:
            p = POWERS[torch.randint(0, len(POWERS), (count,), device=DEVICE)]
            sa = torch.remainder(torch.randint(1, LIMIT, (count,), device=DEVICE), p - 1) + 1
            sb = p - sa + torch.randint(-1, 2, (count,), device=DEVICE)
            ha = torch.randint(1, 9, (count,), device=DEVICE)
            hb = torch.randint(1, 9, (count,), device=DEVICE)
            aa = torch.remainder(ha * (LIMIT // 10) + torch.randint(0, LIMIT // 10, (count,), device=DEVICE), LIMIT)
            bb = torch.remainder(hb * (LIMIT // 10) + torch.randint(0, LIMIT // 10, (count,), device=DEVICE), LIMIT)
            aa = aa - torch.remainder(aa, p) + sa
            bb = bb - torch.remainder(bb, p) + torch.remainder(sb, p)
            aa = aa.clamp(LOW, LIMIT - 1); bb = bb.clamp(LOW, LIMIT - 1)
            a[:n][idx], b[:n][idx] = aa, bb
        # Long runs of zeroes/nines with unrelated high prefixes.
        idx = mode == 2
        count = int(idx.sum())
        if count:
            p = POWERS[torch.randint(0, len(POWERS), (count,), device=DEVICE)]
            aa = torch.randint(LOW, LIMIT, (count,), device=DEVICE)
            bb = torch.randint(LOW, LIMIT, (count,), device=DEVICE)
            aa = aa - torch.remainder(aa, p) + torch.where(torch.rand(count, device=DEVICE) < .5, p - 1, torch.zeros_like(p))
            bb = bb - torch.remainder(bb, p) + torch.where(torch.rand(count, device=DEVICE) < .5, p - 1, torch.zeros_like(p))
            a[:n][idx], b[:n][idx] = aa, bb
        # Decimal boundaries and sparse/repeated edge values.
        idx = mode >= 3
        count = int(idx.sum())
        if count:
            a[:n][idx] = EDGE_VALUES[torch.randint(0, len(EDGE_VALUES), (count,), device=DEVICE)]
            if bool((mode[idx] == 3).any()):
                chosen = b[:n][idx]
                pool = EDGE_VALUES[torch.randint(0, len(EDGE_VALUES), (count,), device=DEVICE)]
                rand = torch.randint(LOW, LIMIT, (count,), device=DEVICE)
                b[:n][idx] = torch.where(mode[idx] == 3, pool, rand)
    result = a + b
    ad, bd, rd = digits(a, 8), digits(b, 8), digits(result, 9)
    tokens = torch.empty((batch, 25), dtype=torch.long, device=DEVICE)
    tokens[:, 0:16:2] = ad
    tokens[:, 1:16:2] = bd
    tokens[:, 16] = 10
    tokens[:, 17:] = rd[:, :8]
    return tokens, rd


def accuracy(model, batches=10, structured=0.0):
    model.eval(); correct = total = 0; min_margin = 1e9
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(4096, structured)
            logits = model(x)[:, 16:25]
            pred = logits.argmax(-1)
            correct += int((pred == y).all(1).sum()); total += len(x)
            top = logits.topk(2, dim=-1).values
            min_margin = min(min_margin, float((top[..., 0] - top[..., 1]).min()))
    model.train()
    return correct / total, min_margin


def train_phase(model, steps, lr, structured, name, warmup=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    for step in range(1, steps + 1):
        if warmup:
            ratio = min(1.0, step / warmup)
            decay = 0.1 + 0.9 * (1 + math.cos(math.pi * max(0, step-warmup) / max(1, steps-warmup))) / 2
            rate = lr * ratio * decay
            for group in optimizer.param_groups: group["lr"] = rate
        x, y = make_batch(BATCH, structured)
        loss = F.cross_entropy(model(x)[:, 16:25].reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 1000 == 0:
            ar, margin = accuracy(model, 4, 0.0)
            ae, _ = accuracy(model, 4, 0.8)
            print(f"{name} {step}/{steps} loss={loss.item():.5f} random={ar:.6f} structured={ae:.6f} margin={margin:.3f}", flush=True)
            torch.save(model.state_dict(), ROOT / f"{name}.pt")


def prune_ffn(model, target_width, module):
    while model.ffn_in.out_features > target_width:
        score = model.ffn_in.weight.norm(dim=1) * model.ffn_out.weight.norm(dim=0)
        remove = int(score.argmin())
        keep = [i for i in range(len(score)) if i != remove]
        smaller = module.AdditionTransformer(model.ffn_in.out_features - 1).to(DEVICE)
        state = model.state_dict()
        state["ffn_in.weight"] = state["ffn_in.weight"][keep]
        state["ffn_out.weight"] = state["ffn_out.weight"][:, keep]
        smaller.load_state_dict(state)
        model = smaller
        print(f"pruned neuron {remove}, now width {model.ffn_in.out_features}", flush=True)
    return model


def export(model):
    path = ROOT / "submission.py"
    text = path.read_text()
    state = model.to("cpu").state_dict()
    entries = []
    for key, value in state.items():
        entries.append(repr(key) + ": torch.tensor(" + repr(value.tolist()) + ")")
    literal = "{\n    " + ",\n    ".join(entries) + "\n}"
    start = text.index("_TRAINED_STATE = ")
    end = text.index("\n\n\ndef build_model", start)
    text = text[:start] + "_TRAINED_STATE = " + literal + text[end:]
    path.write_text(text)
    print(f"exported {sum(v.numel() for v in state.values())} parameters to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(240023)
    module = load_submission()
    model = module.AdditionTransformer(4).to(DEVICE)
    if args.quick:
        train_phase(model, 10, 2e-3, .35, "teacher", 2)
        model = prune_ffn(model, 2, module); export(model); return
    train_phase(model, 36000, 2e-3, .32, "teacher", 1000)
    # Stabilize the teacher before structural changes.
    train_phase(model, 6000, 5e-5, .50, "teacher_stable")
    model = prune_ffn(model, 3, module)
    train_phase(model, 14000, 2e-5, .55, "width3")
    train_phase(model, 5000, 5e-6, .70, "width3_edge")
    model = prune_ffn(model, 2, module)
    train_phase(model, 24000, 2e-5, .55, "width2")
    train_phase(model, 12000, 5e-6, .72, "width2_edge")
    ar, mr = accuracy(model, 100, 0.0); ae, me = accuracy(model, 100, .85)
    print(f"FINAL random={ar:.8f}/{mr:.3f} structured={ae:.8f}/{me:.3f}")
    torch.save(model.state_dict(), ROOT / "final.pt")
    export(model)


if __name__ == "__main__":
    main()
