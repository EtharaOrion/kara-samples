import argparse
import base64
import random
import zlib

import torch
import torch.nn.functional as F

from submission import Adder

LIMIT = 100_000_000_000_000


def batch_data(batch, device, special=0.15):
    a = torch.randint(0, LIMIT, (batch,), device=device)
    b = torch.randint(0, LIMIT, (batch,), device=device)
    if special:
        n = int(batch * special)
        # Full-pair difficult examples: repeated digits and long carry runs.
        rep_a = torch.randint(0, 10, (n,), device=device)
        rep_b = torch.randint(0, 10, (n,), device=device)
        repunit = 11_111_111_111_111
        a[:n] = rep_a * repunit
        b[:n] = rep_b * repunit
        m = n // 3
        if m:
            lengths = torch.randint(2, 14, (m,), device=device)
            powers = torch.tensor([10 ** int(v) for v in lengths.tolist()], device=device)
            a[:m] = powers - torch.randint(1, 10, (m,), device=device)
            b[:m] = torch.randint(1, 10, (m,), device=device)
            alternating = torch.tensor([90_909_090_909_090, 9_090_909_090_909], device=device)
            pick = torch.randint(0, 2, (m,), device=device)
            a[m:2*m] = alternating[pick]
            b[m:2*m] = alternating[1-pick]
    total = a + b
    seq = torch.empty((batch, 45), dtype=torch.long, device=device)
    aa, bb, ss = a, b, total
    for i in range(15):
        if i < 14:
            aa, ad = torch.div(aa, 10, rounding_mode='floor'), aa % 10
            bb, bd = torch.div(bb, 10, rounding_mode='floor'), bb % 10
        else:
            ad = bd = torch.zeros_like(a)
        ss, sd = torch.div(ss, 10, rounding_mode='floor'), ss % 10
        seq[:, 3*i] = ad
        seq[:, 3*i+1] = bd
        seq[:, 3*i+2] = sd
    return seq


def evaluate(model, batches=10, batch=2048):
    model.eval()
    exact = correct = count = 0
    with torch.inference_mode():
        for _ in range(batches):
            x = batch_data(batch, next(model.parameters()).device, special=0)
            pred = model(x)[:, 1::3].argmax(-1)
            target = x[:, 2::3]
            exact += (pred == target).all(1).sum().item()
            correct += (pred == target).sum().item()
            count += batch
    model.train()
    return exact / count, correct / (count * 15)


def export(model, path='/workspace/submission.py'):
    raw = b''.join(p.detach().cpu().half().numpy().tobytes() for p in model.parameters())
    blob = base64.b85encode(zlib.compress(raw, 9)).decode()
    text = open(path).read()
    start = text.index('_WEIGHTS = "') + len('_WEIGHTS = "')
    end = text.index('"', start)
    open(path, 'w').write(text[:start] + blob + text[end:])
    print('exported', len(blob), 'characters')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2600)
    parser.add_argument('--batch', type=int, default=4096)
    args = parser.parse_args()
    torch.manual_seed(47)
    random.seed(47)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Adder().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.005)
    for step in range(1, args.steps + 1):
        if step == 1001:
            for group in optimizer.param_groups: group['lr'] = 1.5e-3
        if step == 1801:
            for group in optimizer.param_groups: group['lr'] = 7e-4
        if step == 2301:
            for group in optimizer.param_groups: group['lr'] = 3e-4
        x = batch_data(args.batch, device, special=0)
        logits = model(x)[:, 1::3]
        loss = F.cross_entropy(logits.reshape(-1, 10), x[:, 2::3].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            exact, digit = evaluate(model, 4)
            print(step, f'loss={loss.item():.6f}', f'tf_exact={exact:.5f}', f'digit={digit:.7f}', flush=True)
    export(model)


if __name__ == '__main__':
    main()

# Continuation utility: loads the fp16 weights embedded by a prior run.
def continue_training(steps=3000, batch=4096, lr=1e-3):
    from submission import build_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = build_model()
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.002)
    for step in range(1, steps + 1):
        if step == 1501:
            for group in optimizer.param_groups: group['lr'] = lr * .4
        if step == 2501:
            for group in optimizer.param_groups: group['lr'] = lr * .15
        x = batch_data(batch, device, special=.10)
        logits = model(x)[:, 1::3]
        loss = F.cross_entropy(logits.reshape(-1, 10), x[:, 2::3].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            exact, digit = evaluate(model, 4)
            print(step, f'loss={loss.item():.6f}', f'tf_exact={exact:.5f}', f'digit={digit:.7f}', flush=True)
        if step % 500 == 0:
            export(model)
    export(model)
