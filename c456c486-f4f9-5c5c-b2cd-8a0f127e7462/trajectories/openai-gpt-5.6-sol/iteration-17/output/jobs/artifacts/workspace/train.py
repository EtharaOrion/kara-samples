import math
import time
import torch
import torch.nn.functional as F
from submission import AdditionTransformer

DEVICE = "cuda"
LIMIT = 100_000_000_000_000
BATCH = 4096
POW10 = torch.tensor([10 ** i for i in range(15)], device=DEVICE, dtype=torch.long)


def digits(values, count=15):
    return (values[:, None] // POW10[None, :count]) % 10


def numbers(ds):
    return (ds * POW10[None, :ds.shape[1]]).sum(1)


def batch_data(batch, structured=True):
    a = torch.randint(LIMIT, (batch,), device=DEVICE)
    b = torch.randint(LIMIT, (batch,), device=DEVICE)
    if structured:
        n = batch // 8
        # Regenerated carry chains at arbitrary positions and lengths.
        da = digits(a[:n], 14)
        db = digits(b[:n], 14)
        start = torch.randint(0, 14, (n,), device=DEVICE)
        length = torch.randint(1, 15, (n,), device=DEVICE)
        end = torch.minimum(start + length, torch.full_like(start, 14))
        p = torch.arange(14, device=DEVICE)[None]
        chain = (p >= start[:, None]) & (p < end[:, None])
        da[chain] = 9
        db[chain] = 0
        rows = torch.arange(n, device=DEVICE)
        db[rows, start] = 1
        a[:n], b[:n] = numbers(da), numbers(db)

        # Repeated operands, sparse increments, complements, and overflow.
        lo, hi = n, 2 * n
        repeated = torch.randint(10, (n, 1), device=DEVICE).expand(-1, 14)
        a[lo:hi] = numbers(repeated)
        sparse_pos = torch.randint(14, (n,), device=DEVICE)
        b[lo:hi] = POW10[sparse_pos] * torch.randint(1, 10, (n,), device=DEVICE)

        lo, hi = 2 * n, 3 * n
        mode = torch.arange(n, device=DEVICE) % 2 == 0
        comp = LIMIT - 1 - a[lo:hi]
        near_max = LIMIT - 1 - torch.randint(1_000_000, (n,), device=DEVICE)
        a[lo:hi] = torch.where(mode, a[lo:hi], near_max)
        b[lo:hi] = torch.where(mode, comp, torch.randint(1, 1_000_000, (n,), device=DEVICE))
    out = a + b
    seq = torch.empty((batch, 45), dtype=torch.long, device=DEVICE)
    seq[:, 0] = 10
    seq[:, 1:29:2] = digits(a, 14)
    seq[:, 2:29:2] = digits(b, 14)
    seq[:, 29] = 11
    seq[:, 30:45] = digits(out, 15)
    return seq


@torch.no_grad()
def validate(model, count=65536, structured=False, chunk=8192):
    model.eval()
    errors = 0
    digit_errors = 0
    for _ in range(math.ceil(count / chunk)):
        size = min(chunk, count)
        count -= size
        full = batch_data(size, structured)
        seq = full[:, :30]
        expected = full[:, 30:]
        for _ in range(15):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat((seq, nxt), 1)
        wrong = seq[:, 30:] != expected
        errors += wrong.any(1).sum().item()
        digit_errors += wrong.sum().item()
    model.train()
    return errors, digit_errors


def main():
    torch.manual_seed(1701)
    torch.set_float32_matmul_precision("high")
    model = AdditionTransformer().to(DEVICE)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98), weight_decay=0.005, fused=True)
    compiled = model
    best = 10**9
    start = time.time()
    total_steps = 26000
    for step in range(1, total_steps + 1):
        if step == 12001:
            for g in opt.param_groups: g["lr"] = 1e-3
        if step == 19001:
            for g in opt.param_groups: g["lr"] = 3e-4
        if step == 23501:
            for g in opt.param_groups: g["lr"] = 1e-4
        full = batch_data(BATCH, structured=(step > 1500))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = compiled(full[:, :-1])[:, 29:44]
            loss = F.cross_entropy(logits.reshape(-1, 10), full[:, 30:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(step, f"loss={loss.item():.6g}", f"lr={opt.param_groups[0]['lr']:.1g}", f"sec={time.time()-start:.1f}", flush=True)
        if step >= 12000 and step % 2000 == 0:
            err, de = validate(model, 32768, False)
            serr, sde = validate(model, 32768, True)
            score = err + serr
            print("validation", err, de, serr, sde, flush=True)
            if score <= best:
                best = score
                torch.save(model.state_dict(), "/workspace/model.pt")
                print("saved", score, flush=True)
    torch.save(model.state_dict(), "/workspace/model_final.pt")


if __name__ == "__main__":
    main()
