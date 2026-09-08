import argparse
import base64
import io
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from submission import AdditionTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOW, HIGH = 10_000_000, 99_999_999


def digits_lsd(values, count):
    powers = 10 ** torch.arange(count, device=values.device)
    return (values[:, None] // powers) % 10


def make_batch(batch, structured=0.2):
    a = torch.randint(LOW, HIGH + 1, (batch,), device=DEVICE)
    b = torch.randint(LOW, HIGH + 1, (batch,), device=DEVICE)
    n = int(batch * structured)
    if n:
        kinds = torch.randint(0, 5, (n,), device=DEVICE)
        x = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
        # Long carries into 100,000,000.
        m = kinds == 0
        a[:n] = torch.where(m, x, a[:n])
        b[:n] = torch.where(m, 100_000_000 - x, b[:n])
        # Sparse round numbers, including difficult zero-heavy operands.
        m = kinds == 1
        places = 10 ** torch.randint(1, 8, (n,), device=DEVICE)
        aa = ((x // places) * places).clamp(LOW, HIGH)
        bb = (torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000).clamp(max=HIGH)
        a[:n] = torch.where(m, aa, a[:n])
        b[:n] = torch.where(m, bb, b[:n])
        # Repeated-digit operands.
        m = kinds == 2
        ra = torch.randint(1, 10, (n,), device=DEVICE) * 11_111_111
        rb = torch.randint(1, 10, (n,), device=DEVICE) * 11_111_111
        a[:n] = torch.where(m, ra, a[:n])
        b[:n] = torch.where(m, rb, b[:n])
        # Carry-biased complementary suffixes.
        m = kinds == 3
        k = torch.randint(1, 8, (n,), device=DEVICE)
        p = 10 ** k
        base = torch.randint(LOW, HIGH + 1, (n,), device=DEVICE)
        suffix = p - base % p
        prefix = torch.randint(1, 10, (n,), device=DEVICE) * 10_000_000
        partner = (prefix + suffix).clamp(LOW, HIGH)
        a[:n] = torch.where(m, base, a[:n])
        b[:n] = torch.where(m, partner, b[:n])
        # Boundary-heavy random values.
        m = kinds == 4
        offsets = torch.randint(0, 100_000, (n,), device=DEVICE)
        hi = torch.where(torch.rand(n, device=DEVICE) < .5,
                         LOW + offsets, HIGH - offsets)
        a[:n] = torch.where(m, hi, a[:n])
        b[:n] = torch.where(m, x, b[:n])
    da, db = digits_lsd(a, 8), digits_lsd(b, 8)
    operands = torch.stack((da, db), dim=2).reshape(batch, 16)
    answer = digits_lsd(a + b, 9)
    inp = torch.cat((operands, torch.full((batch, 1), 10, device=DEVICE), answer[:, :-1]), dim=1)
    return inp.long(), answer.long(), a, b


@torch.no_grad()
def evaluate(model, batches=10, batch=1000, structured=0.0):
    model.eval()
    correct = total = 0
    digit_correct = 0
    for _ in range(batches):
        inp, target, _, _ = make_batch(batch, structured)
        seq = inp[:, :17]
        predictions = []
        for _ in range(9):
            pred = model(seq)[:, -1].argmax(1)
            predictions.append(pred)
            seq = torch.cat((seq, pred[:, None]), 1)
        pred = torch.stack(predictions, 1)
        correct += (pred == target).all(1).sum().item()
        digit_correct += (pred == target).sum().item()
        total += batch
    model.train()
    return correct / total, digit_correct / (total * 9)


def write_submission(model):
    buffer = io.BytesIO()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, buffer)
    encoded = base64.b85encode(buffer.getvalue()).decode("ascii")
    path = Path("/workspace/submission.py")
    text = path.read_text()
    start = text.index('_TRAINED_STATE = "')
    end = text.index('"\n\n\ndef build_model', start)
    text = text[:start] + '_TRAINED_STATE = "' + encoded + text[end:]
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = AdditionTransformer().to(DEVICE)
    checkpoint = Path("/workspace/model.pt")
    if args.resume and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=5e-5)
    model.train()
    for step in range(1, args.steps + 1):
        structured = 0.25 if step > args.steps // 2 else 0.15
        inp, target, _, _ = make_batch(args.batch, structured)
        logits = model(inp)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 1000 == 0 or step == 1:
            random_acc, digits = evaluate(model, 3, 1000, 0.0)
            edge_acc, _ = evaluate(model, 2, 1000, 1.0)
            print(step, f"loss={loss.item():.5f}", f"random={random_acc:.4f}",
                  f"digit={digits:.5f}", f"edge={edge_acc:.4f}", flush=True)
            torch.save(model.state_dict(), checkpoint)
            write_submission(model)
    random_acc, digits = evaluate(model, 20, 1000, 0.0)
    edge_acc, _ = evaluate(model, 10, 1000, 1.0)
    print("FINAL", random_acc, digits, edge_acc)
    torch.save(model.state_dict(), checkpoint)
    write_submission(model)


if __name__ == "__main__":
    main()
