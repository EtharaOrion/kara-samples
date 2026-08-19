from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
import train


def boundary_batch(n, device):
    a = torch.randint(0, train.LIMIT, (n,), device=device)
    b = torch.randint(0, train.LIMIT, (n,), device=device)
    # Contrast true carries with near-identical non-carries around every power of ten.
    k = torch.randint(1, 15, (n,), device=device)
    p = torch.pow(torch.tensor(10, dtype=torch.int64, device=device), k)
    delta = torch.randint(-40, 41, (n,), device=device)
    small = torch.randint(0, 41, (n,), device=device)
    base = (p + delta).clamp(0, train.LIMIT - 1)
    # Add arbitrary higher prefixes when room remains, preserving the local boundary.
    room = (train.LIMIT - 1 - base) // p
    prefix_units = (torch.rand(n, device=device) * (room + 1).float()).long()
    a = base + prefix_units * p
    b = small
    swap = torch.rand(n, device=device) < 0.5
    aa = torch.where(swap, b, a)
    bb = torch.where(swap, a, b)
    return train.to_digits(aa), train.to_digits(bb), train.to_digits(aa + bb)


def mixed_batch(n, device):
    x1, b1, y1 = train.make_batch(n // 2, device, 0.4)
    x2, b2, y2 = boundary_batch(n - n // 2, device)
    return torch.cat((x1, x2)), torch.cat((b1, b2)), torch.cat((y1, y2))


def main():
    torch.manual_seed(20250815)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    train.a_device = device
    model = train.Adder().to(device)
    model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.001)
    best_score = 10**9
    for step in range(1, 6001):
        if step == 4000:
            for g in opt.param_groups: g['lr'] = 1e-5
        a, b, y = mixed_batch(8192, device)
        loss = F.cross_entropy(model(a, b).reshape(-1, 10), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            er, nr = train.evaluate(model, 16, 4096, 0.0)
            es, ns = train.evaluate(model, 8, 4096, 0.75)
            with torch.no_grad():
                ea = 0
                for _ in range(8):
                    x, z, yy = boundary_batch(4096, device)
                    ea += (model(x,z).argmax(-1) != yy).any(1).sum().item()
            score = er * 4 + es * 2 + ea
            print(step, float(loss), 'random', er, nr, 'structured', es, ns, 'boundary', ea, 32768, flush=True)
            if score <= best_score:
                best_score = score
                torch.save(model.state_dict(), '/workspace/best.pt')
                train.export(model, Path('/workspace/submission.py'))
    model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
    train.export(model, Path('/workspace/submission.py'))

if __name__ == '__main__':
    main()
