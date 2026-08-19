import math
import random
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
from submission import AdditionTransformer

DEVICE = 'cuda'
BATCH = 8192
LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10 ** i for i in range(15)], device=DEVICE, dtype=torch.long)

def digits(x, n):
    return (x[:, None] // POW10[:n]) % 10

def batch_data(batch, structured=0.20):
    a = torch.randint(0, LIMIT, (batch,), device=DEVICE)
    b = torch.randint(0, LIMIT, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        # Regenerated complete-pair families: sparse increments, repeated operands,
        # complements, near-boundary values, and randomized carry chains.
        q = n // 5
        idx = torch.arange(q, device=DEVICE)
        a[:q] = torch.randint(0, LIMIT, (q,), device=DEVICE)
        b[:q] = 10 ** torch.randint(0, 14, (q,), device=DEVICE)
        d = torch.randint(0, 10, (q,), device=DEVICE)
        rep = sum(d * (10 ** i) for i in range(14))
        a[q:2*q] = rep
        b[q:2*q] = torch.randint(0, LIMIT, (q,), device=DEVICE)
        x = torch.randint(0, LIMIT, (q,), device=DEVICE)
        a[2*q:3*q] = x
        b[2*q:3*q] = (LIMIT - 1) - x
        a[3*q:4*q] = (LIMIT - 1) - torch.randint(0, 100000, (q,), device=DEVICE)
        b[3*q:4*q] = torch.randint(0, 100000, (q,), device=DEVICE)
        length = torch.randint(1, 15, (n-4*q,), device=DEVICE)
        start = torch.minimum(torch.randint(0, 14, (n-4*q,), device=DEVICE), 14-length)
        p = POW10[start]
        run = (POW10[length] - 1) * p
        low = torch.randint(0, LIMIT, (n-4*q,), device=DEVICE)
        a[4*q:n] = (low // (run + 1) * (run + 1) + run).remainder(LIMIT)
        b[4*q:n] = p
    source = torch.cat((digits(a, 14), digits(b, 14)), 1)
    target = digits(a + b, 15)
    previous = torch.cat((torch.full((batch, 1), 10, device=DEVICE), target[:, :-1]), 1)
    return source, previous, target

@torch.no_grad()
def evaluate(model, batches=8, batch=8192, structured=0.0):
    model.eval()
    bad = total = 0
    for _ in range(batches):
        source, _, target = batch_data(batch, structured)
        memory = model.encode(source)
        generated = torch.full((batch, 1), 10, dtype=torch.long, device=DEVICE)
        for i in range(15):
            x = model.token(generated) + model.out_place[:generated.shape[1]]
            for decoder in model.decoders:
                x = decoder(x, memory)
            logits = model.head(model.norm(x))
            generated = torch.cat((generated, logits[:, -1].argmax(-1, keepdim=True)), 1)
        bad += (generated[:, 1:] != target).any(1).sum().item()
        total += batch
    model.train()
    return bad, total

def export(model, path):
    flat = torch.cat([p.detach().cpu().float().reshape(-1) for p in model.parameters()]).tolist()
    base = Path('/workspace/submission.py').read_text()
    marker = '# Training replaces this seed initialization with trained parameter values.'
    prefix = base.split(marker)[0]
    values = repr(flat)
    tail = f'''TRAINED = {values}\n\ndef build_model():\n    model = AdditionTransformer()\n    values = torch.tensor(TRAINED, dtype=torch.float32)\n    offset = 0\n    with torch.no_grad():\n        for parameter in model.parameters():\n            count = parameter.numel()\n            parameter.copy_(values[offset:offset + count].view_as(parameter))\n            offset += count\n    model.eval()\n    return model, {{"architecture": "separated encoder-decoder transformer", "digits": 14, "training": "full random pairs and regenerated structured pairs"}}\n\ndef add(model, a: int, b: int) -> int:\n    left = [ord(c) - 48 for c in f"{{a:014d}}"[::-1]]\n    right = [ord(c) - 48 for c in f"{{b:014d}}"[::-1]]\n    source = torch.tensor([left + right], dtype=torch.long, device=next(model.parameters()).device)\n    generated = torch.full((1, 1), 10, dtype=torch.long, device=source.device)\n    with torch.no_grad():\n        memory = model.encode(source)\n        for _ in range(15):\n            x = model.token(generated) + model.out_place[:generated.shape[1]]\n            for decoder in model.decoders:\n                x = decoder(x, memory)\n            logits = model.head(model.norm(x))\n            digit = logits[:, -1].argmax(-1, keepdim=True)\n            generated = torch.cat((generated, digit), 1)\n    text = ''.join(chr(48 + int(x)) for x in generated[0, 1:].flip(0))\n    return int(text)\n'''
    Path(path).write_text(prefix + tail)

def main():
    torch.manual_seed(19019)
    torch.set_float32_matmul_precision('high')
    model = AdditionTransformer().to(DEVICE)
    raw = model
    opt = torch.optim.AdamW(raw.parameters(), lr=3e-3, weight_decay=.003)
    best = None
    for step in range(1, 26001):
        if step == 10001:
            for g in opt.param_groups: g['lr'] = 1e-3
        if step == 18001:
            for g in opt.param_groups: g['lr'] = 3e-4
        if step == 23001:
            for g in opt.param_groups: g['lr'] = 1e-4
        source, previous, target = batch_data(BATCH, .20 if step < 18000 else .35)
        logits = raw(source, previous)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            bad, total = evaluate(raw, 2, 4096, 0.0)
            sbad, stotal = evaluate(raw, 2, 4096, .8)
            print(step, float(loss), f'random {bad}/{total}', f'struct {sbad}/{stotal}', flush=True)
            if bad + sbad <= 2:
                best = {k:v.detach().cpu().clone() for k,v in raw.state_dict().items()}
                export(raw, '/workspace/submission.py')
    if best is not None:
        raw.load_state_dict(best)
    export(raw, '/workspace/submission.py')
    bad, total = evaluate(raw, 16, 8192, 0.0)
    sbad, stotal = evaluate(raw, 16, 8192, .8)
    print('FINAL', bad, total, sbad, stotal, 'params', sum(p.numel() for p in raw.parameters()))

if __name__ == '__main__':
    main()
