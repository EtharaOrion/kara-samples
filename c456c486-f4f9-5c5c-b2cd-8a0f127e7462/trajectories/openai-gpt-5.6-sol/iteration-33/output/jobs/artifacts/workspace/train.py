import argparse
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("submission_arch", ROOT / "submission.py")
SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMISSION)
AdditionTransformer = SUBMISSION.AdditionTransformer

LIMIT = 100_000_000_000_000
POWERS = torch.tensor([10 ** i for i in range(15)], dtype=torch.long)


def digits(values, places=15):
    powers = POWERS[:places].to(values.device)
    return (values[:, None] // powers[None, :]) % 10


def values(ds):
    powers = POWERS[:14].to(ds.device)
    return (ds * powers).sum(dim=1)


def patterned_digits(n, device, kind=None):
    a = torch.randint(0, 10, (n, 14), device=device)
    b = torch.randint(0, 10, (n, 14), device=device)
    kinds = torch.randint(0, 7, (n,), device=device) if kind is None else torch.full((n,), kind, device=device)
    pos = torch.randint(0, 14, (n,), device=device)
    length = torch.randint(1, 15, (n,), device=device)
    cols = torch.arange(14, device=device)[None, :]
    run = (cols >= pos[:, None]) & (cols < (pos + length).clamp(max=14)[:, None])

    # 0: a run of nines plus an increment; 1: matched non-carry run.
    m = kinds == 0
    a = torch.where(m[:, None] & run, torch.full_like(a, 9), a)
    b = torch.where(m[:, None] & run, torch.zeros_like(b), b)
    b[m, pos[m]] = 1
    m = kinds == 1
    a = torch.where(m[:, None] & run, torch.full_like(a, 8), a)
    b = torch.where(m[:, None] & run, torch.zeros_like(b), b)
    b[m, pos[m]] = 1

    # 2: complementary run with carry entering at its first column.
    m = kinds == 2
    comp = 9 - a
    b = torch.where(m[:, None] & run, comp, b)
    a[m, pos[m]] = 9
    b[m, pos[m]] = 1

    # 3: sparse operands; half the time an exact isolated 5+5 boundary.
    m = kinds == 3
    a[m] = 0
    b[m] = 0
    rows = torch.where(m)[0]
    if rows.numel():
        p = pos[m]
        use_five = torch.rand(rows.numel(), device=device) < 0.5
        da = torch.randint(1, 10, (rows.numel(),), device=device)
        db = torch.randint(1, 10, (rows.numel(),), device=device)
        da[use_five] = 5
        db[use_five] = 5
        a[rows, p] = da
        b[rows, p] = db

    # 4: isolated high-column 9+9, including the top source column.
    m = kinds == 4
    a[m] = 0
    b[m] = 0
    rows = torch.where(m)[0]
    if rows.numel():
        p = pos[m]
        a[rows, p] = 9
        b[rows, p] = 9

    # 5: repeated digits; 6: blocks with abrupt boundaries.
    m = kinds == 5
    if m.any():
        da = torch.randint(0, 10, (int(m.sum()), 1), device=device)
        db = torch.randint(0, 10, (int(m.sum()), 1), device=device)
        a[m] = da.expand(-1, 14)
        b[m] = db.expand(-1, 14)
    m = kinds == 6
    if m.any():
        cut = pos[m, None]
        c = cols.expand(int(m.sum()), -1)
        lo_a = torch.randint(0, 10, (int(m.sum()), 1), device=device)
        hi_a = torch.randint(0, 10, (int(m.sum()), 1), device=device)
        lo_b = torch.randint(0, 10, (int(m.sum()), 1), device=device)
        hi_b = torch.randint(0, 10, (int(m.sum()), 1), device=device)
        a[m] = torch.where(c < cut, lo_a, hi_a)
        b[m] = torch.where(c < cut, lo_b, hi_b)
    return a, b


def batch_data(batch, device, structured_fraction, boundary_fraction=0.0):
    n_struct = int(batch * structured_fraction)
    n_boundary = int(batch * boundary_fraction)
    n_random = batch - n_struct - n_boundary
    av = torch.randint(0, LIMIT, (n_random,), device=device)
    bv = torch.randint(0, LIMIT, (n_random,), device=device)
    ad = digits(av, 14)
    bd = digits(bv, 14)
    if n_struct:
        sa, sb = patterned_digits(n_struct, device)
        ad = torch.cat((ad, sa))
        bd = torch.cat((bd, sb))
    if n_boundary:
        ba, bb = patterned_digits(n_boundary, device, kind=4)
        half = n_boundary // 2
        if half:
            five_a, five_b = patterned_digits(half, device, kind=3)
            ba[:half] = five_a
            bb[:half] = five_b
        ad = torch.cat((ad, ba))
        bd = torch.cat((bd, bb))
    sums = values(ad) + values(bd)
    target = digits(sums, 15)
    pad = torch.full((batch, 1), 10, dtype=torch.long, device=device)
    source_a = torch.cat((ad, pad), dim=1)
    source_b = torch.cat((bd, pad), dim=1)
    prefix = torch.cat((pad, target[:, :-1]), dim=1)
    return source_a, source_b, prefix, target


@torch.no_grad()
def autoregressive_errors(model, batches, batch_size, generator):
    errors = 0
    total = 0
    for _ in range(batches):
        aa, bb, _, target = generator(batch_size)
        prefix = torch.full((batch_size, 1), 10, dtype=torch.long, device=aa.device)
        predictions = []
        for _ in range(15):
            logits = model(aa, bb, prefix)
            nxt = logits[:, -1].argmax(dim=-1)
            predictions.append(nxt)
            prefix = torch.cat((prefix, nxt[:, None]), dim=1)
        pred = torch.stack(predictions, dim=1)
        errors += int((pred != target).any(dim=1).sum())
        total += batch_size
    return errors, total


def export_submission(model):
    path = ROOT / "submission.py"
    text = path.read_text()
    marker = "_STATE = None"
    state = {k: v.detach().cpu().float().tolist() for k, v in model.state_dict().items()}
    replacement = "_STATE = " + repr(state)
    if marker not in text:
        start = text.index("_STATE = {")
        end = text.index("\n\n\ndef build_model", start)
        text = text[:start] + replacement + text[end:]
    else:
        text = text.replace(marker, replacement)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=38000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=3301)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = AdditionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.003, fused=True)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)

    best_state = None
    best_score = math.inf
    for step in range(1, args.steps + 1):
        if step <= 12000:
            lr, sf, bf = 3e-3, 0.30, 0.05
        elif step <= 24000:
            lr, sf, bf = 1e-3, 0.36, 0.08
        elif step <= 32000:
            lr, sf, bf = 3e-4, 0.40, 0.12
        else:
            lr, sf, bf = 1e-4, 0.38, 0.20
        optimizer.param_groups[0]["lr"] = lr
        aa, bb, prefix, target = batch_data(args.batch_size, device, sf, bf)
        logits = model(aa, bb, prefix)[:, 15:]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0 or step == args.steps:
            model.eval()
            e1, n1 = autoregressive_errors(model, 4, 4096, lambda n: batch_data(n, device, 0.0))
            e2, n2 = autoregressive_errors(model, 4, 4096, lambda n: batch_data(n, device, 0.55, 0.20))
            score = e1 * 4 + e2
            print(f"step {step} loss {loss.item():.6g} random {e1}/{n1} mixed {e2}/{n2}", flush=True)
            if step >= 16000 and score <= best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ROOT / "best.pt")
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    export_submission(model)
    torch.save(model.state_dict(), ROOT / "final.pt")
    print("exported", ROOT / "submission.py", "best_score", best_score, flush=True)


if __name__ == "__main__":
    main()
