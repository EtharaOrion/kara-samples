import argparse
import importlib.util
import pathlib
import random

import torch
import torch.nn.functional as F


ROOT = pathlib.Path("/workspace")
spec = importlib.util.spec_from_file_location("submission", ROOT / "submission.py")
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


def batch(size, device, generator):
    a = torch.randint(10_000_000, 100_000_000, (size,), device=device, generator=generator)
    b = torch.randint(10_000_000, 100_000_000, (size,), device=device, generator=generator)
    powers8 = 10 ** torch.arange(8, device=device)
    ad = (a[:, None] // powers8) % 10
    bd = (b[:, None] // powers8) % 10
    operands = torch.stack((ad, bd), dim=2).reshape(size, 16)
    powers9 = 10 ** torch.arange(9, device=device)
    result = ((a + b)[:, None] // powers9) % 10
    x = torch.empty((size, 25), dtype=torch.long, device=device)
    x[:, :16] = operands
    x[:, 16] = 10
    x[:, 17:] = result[:, :-1]
    return x, result


@torch.no_grad()
def validate(model, batches=20, size=2048, seed=8675309):
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    exact = digits = total = 0
    for _ in range(batches):
        x, target = batch(size, device, generator)
        tokens = x[:, :17]
        predictions = []
        for _ in range(9):
            prediction = model(tokens)[:, -1].argmax(1)
            predictions.append(prediction)
            tokens = torch.cat((tokens, prediction[:, None]), 1)
        prediction = torch.stack(predictions, 1)
        exact += (prediction == target).all(1).sum().item()
        digits += (prediction == target).sum().item()
        total += size
    model.train()
    return exact / total, digits / (total * 9)


def write_submission(model, template):
    state = {key: value.detach().float().cpu() for key, value in model.state_dict().items()}
    marker = "# Learned parameters produced by /workspace/train.py."
    lines = [template.split(marker, 1)[0], marker + "\n", "_TRAINED_STATE = {\n"]
    for key, value in state.items():
        lines.append(f"    {key!r}: torch.tensor({value.tolist()!r}),\n")
    lines.append("}\n\n")
    suffix = template.split("def build_model():", 1)[1]
    lines.append("def build_model():" + suffix)
    (ROOT / "submission.py").write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = submission.AdditionTransformer(args.width, args.heads, args.hidden, args.layers).to(device)
    print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr / 20)
    generator = torch.Generator(device=device).manual_seed(args.seed + 100)
    model.train()
    for step in range(1, args.steps + 1):
        x, target = batch(args.batch, device, generator)
        logits = model(x)[:, 16:25]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 500 == 0:
            exact, digit = validate(model, batches=4)
            print(f"step={step} loss={loss.item():.5f} exact={exact:.5%} digit={digit:.5%}", flush=True)
    exact, digit = validate(model, batches=50)
    print(f"final exact={exact:.6%} digit={digit:.6%}")
    if args.candidate:
        torch.save(model.state_dict(), ROOT / "candidate.pt")
    else:
        template = (ROOT / "submission.py").read_text()
        write_submission(model, template)
        torch.save(model.state_dict(), ROOT / "model.pt")


if __name__ == "__main__":
    main()
