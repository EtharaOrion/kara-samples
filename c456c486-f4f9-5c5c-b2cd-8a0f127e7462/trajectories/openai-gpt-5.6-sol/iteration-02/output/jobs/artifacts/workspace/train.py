import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from submission import AdditionTransformer, DIGITS


def batch_data(batch, device):
    a = torch.randint(0, 10**14, (batch,), device=device)
    b = torch.randint(0, 10**14, (batch,), device=device)
    powers = 10 ** torch.arange(DIGITS, device=device)
    ad = (a[:, None] // powers) % 10
    bd = (b[:, None] // powers) % 10
    total = a + b
    target = (total[:, None] // powers) % 10
    sequence = torch.stack((ad, bd, target), dim=2).reshape(batch, 3 * DIGITS)
    return sequence.long(), target.long()


def accuracy(model, batches=10, batch=2048):
    good = count = 0
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            teacher, target = batch_data(batch, 'cuda')
            tokens = torch.empty((batch, 0), dtype=torch.long, device='cuda')
            out = []
            for i in range(DIGITS):
                tokens = torch.cat((tokens, teacher[:, 3*i:3*i+2]), 1)
                digit = model(tokens)[:, -1].argmax(1)
                out.append(digit)
                tokens = torch.cat((tokens, digit[:, None]), 1)
            pred = torch.stack(out, 1)
            good += (pred == target).all(1).sum().item()
            count += batch
    model.train()
    return good / count


def export(model, path):
    source = Path(path).read_text()
    marker = '# Replaced by train.py after optimization.\n_WEIGHTS = None'
    weights = {k: v.detach().float().cpu().reshape(-1).tolist() for k, v in model.state_dict().items()}
    replacement = '# Trained from scratch on random full-length operand pairs.\n_WEIGHTS = ' + repr(weights)
    source = source.replace(marker, replacement)
    Path(path).write_text(source)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-3)
    args = p.parse_args()
    torch.manual_seed(48272)
    torch.set_float32_matmul_precision('high')
    model = AdditionTransformer().cuda().train()
    checkpoint = Path('/workspace/model.pt')
    if checkpoint.exists(): model.load_state_dict(torch.load(checkpoint, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    positions = torch.arange(1, 3 * DIGITS, 3, device='cuda')
    for step in range(1, args.steps + 1):
        sequence, target = batch_data(args.batch, 'cuda')
        logits = model(sequence)[:, positions]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            torch.save(model.state_dict(), '/workspace/model.pt')
            acc = accuracy(model, 1, 256)
            print(step, f'loss={loss.item():.6f}', f'exact={acc:.4f}', flush=True)
            if acc >= .999:
                break
    torch.save(model.state_dict(), '/workspace/model.pt')
    print('validation', accuracy(model, 10, 1024), flush=True)
    export(model, '/workspace/submission.py')


if __name__ == '__main__':
    main()
