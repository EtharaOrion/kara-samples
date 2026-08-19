import argparse
import importlib.util
import math
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path("/workspace")
MAX_VALUE = 100_000_000_000_000
DEVICE = torch.device("cuda")
POWERS = (10 ** torch.arange(15, device=DEVICE, dtype=torch.int64))


def load_architecture():
    spec = importlib.util.spec_from_file_location("submission", ROOT / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_digits(values, count=15):
    return (values[:, None] // POWERS[None, :count]) % 10


def from_digits(digits):
    return (digits * POWERS[None, :digits.shape[1]]).sum(dim=1)


def carry_examples(count):
    a = torch.randint(0, 10, (count, 14), device=DEVICE)
    b = torch.randint(0, 10, (count, 14), device=DEVICE)
    start = torch.randint(0, 14, (count, 1), device=DEVICE)
    available = 14 - start
    length = 1 + (torch.rand((count, 1), device=DEVICE) * available).long()
    positions = torch.arange(14, device=DEVICE)[None, :]
    trigger = positions == start
    continuation = (positions > start) & (positions < start + length)
    terminator = positions == start + length

    trigger_a = torch.randint(1, 10, (count, 14), device=DEVICE)
    trigger_b = 10 - trigger_a + (torch.rand((count, 14), device=DEVICE) * trigger_a).long()
    run_a = torch.randint(0, 10, (count, 14), device=DEVICE)
    run_b = 9 - run_a
    stop_a = torch.randint(0, 9, (count, 14), device=DEVICE)
    stop_b = (torch.rand((count, 14), device=DEVICE) * (9 - stop_a)).long()
    a = torch.where(trigger, trigger_a, a)
    b = torch.where(trigger, trigger_b, b)
    a = torch.where(continuation, run_a, a)
    b = torch.where(continuation, run_b, b)
    a = torch.where(terminator, stop_a, a)
    b = torch.where(terminator, stop_b, b)
    return from_digits(a), from_digits(b)


def noncarry_examples(count):
    a = torch.randint(0, 10, (count, 14), device=DEVICE)
    ceiling = 10 - a
    b = (torch.rand((count, 14), device=DEVICE) * ceiling).long()
    start = torch.randint(0, 14, (count, 1), device=DEVICE)
    available = 14 - start
    length = 1 + (torch.rand((count, 1), device=DEVICE) * available).long()
    positions = torch.arange(14, device=DEVICE)[None, :]
    run = (positions >= start) & (positions < start + length)
    run_a = torch.randint(0, 10, (count, 14), device=DEVICE)
    a = torch.where(run, run_a, a)
    b = torch.where(run, 9 - run_a, b)
    return from_digits(a), from_digits(b)


def patterned_examples(count):
    family = torch.arange(count, device=DEVICE) % 5
    a = torch.randint(0, MAX_VALUE, (count,), device=DEVICE)
    b = torch.randint(0, MAX_VALUE, (count,), device=DEVICE)

    repeated_a = torch.randint(0, 10, (count,), device=DEVICE) * 11_111_111_111_111
    repeated_b = torch.randint(0, 10, (count,), device=DEVICE) * 11_111_111_111_111
    a = torch.where(family == 0, repeated_a, a)
    b = torch.where(family == 0, repeated_b, b)

    exponent = torch.randint(1, 15, (count,), device=DEVICE)
    boundary = POWERS[exponent] - 1
    small = torch.randint(0, 11, (count,), device=DEVICE)
    swap = torch.rand(count, device=DEVICE) < 0.5
    sparse_a = torch.where(swap, boundary, small)
    sparse_b = torch.where(swap, small, boundary)
    a = torch.where(family == 1, sparse_a, a)
    b = torch.where(family == 1, sparse_b, b)

    low = torch.randint(1, MAX_VALUE, (count,), device=DEVICE)
    complement = MAX_VALUE - low
    a = torch.where(family == 2, low, a)
    b = torch.where(family == 2, complement, b)

    digit_a = torch.randint(0, 10, (count, 14), device=DEVICE)
    digit_b = 9 - digit_a
    contrast_a = from_digits(digit_a)
    contrast_b = from_digits(digit_b)
    a = torch.where(family == 3, contrast_a, a)
    b = torch.where(family == 3, contrast_b, b)

    top = torch.randint(1, 10, (count,), device=DEVICE) * POWERS[13]
    tail = torch.randint(0, 1000, (count,), device=DEVICE)
    a = torch.where(family == 4, top - 1, a)
    b = torch.where(family == 4, tail, b)
    return a, b


def make_batch(batch_size, structured_fraction=0.25):
    a = torch.randint(0, MAX_VALUE, (batch_size,), device=DEVICE)
    b = torch.randint(0, MAX_VALUE, (batch_size,), device=DEVICE)
    structured = int(batch_size * structured_fraction)
    section = structured // 3
    if section:
        a[:section], b[:section] = carry_examples(section)
        a[section:2 * section], b[section:2 * section] = noncarry_examples(section)
        a[2 * section:structured], b[2 * section:structured] = patterned_examples(structured - 2 * section)
    answer = a + b
    a_digits = to_digits(a, 14)
    b_digits = to_digits(b, 14)
    sentinel = torch.full((batch_size, 1), 10, dtype=torch.long, device=DEVICE)
    a_tokens = torch.cat((a_digits, sentinel), dim=1)
    b_tokens = torch.cat((b_digits, sentinel), dim=1)
    targets = to_digits(answer, 15)
    return a_tokens, b_tokens, targets


def loss_for(model, batch_size, structured_fraction):
    a, b, targets = make_batch(batch_size, structured_fraction)
    logits = model(a, b, targets[:, :-1])[:, 14:]
    return F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))


@torch.inference_mode()
def evaluate(model, batches, batch_size, generator="mixed"):
    errors = 0
    examples = 0
    digit_errors = 0
    for _ in range(batches):
        if generator == "uniform":
            a_values = torch.randint(0, MAX_VALUE, (batch_size,), device=DEVICE)
            b_values = torch.randint(0, MAX_VALUE, (batch_size,), device=DEVICE)
        elif generator == "carry":
            a_values, b_values = carry_examples(batch_size)
        elif generator == "noncarry":
            a_values, b_values = noncarry_examples(batch_size)
        else:
            a_values, b_values = patterned_examples(batch_size)
        targets = to_digits(a_values + b_values, 15)
        sentinel = torch.full((batch_size, 1), 10, dtype=torch.long, device=DEVICE)
        a_tokens = torch.cat((to_digits(a_values, 14), sentinel), dim=1)
        b_tokens = torch.cat((to_digits(b_values, 14), sentinel), dim=1)
        generated = torch.empty((batch_size, 0), dtype=torch.long, device=DEVICE)
        for _position in range(15):
            logits = model(a_tokens, b_tokens, generated)
            generated = torch.cat((generated, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        mismatches = generated != targets
        errors += mismatches.any(dim=1).sum().item()
        digit_errors += mismatches.sum().item()
        examples += batch_size
    return errors, digit_errors, examples


def systematic_cases():
    pairs = {(0, 0), (MAX_VALUE - 1, 0), (MAX_VALUE - 1, MAX_VALUE - 1),
             (MAX_VALUE - 1, 1), (1, MAX_VALUE - 1)}
    for power in (10 ** k for k in range(1, 15)):
        for delta in range(-5, 6):
            value = power + delta
            if 0 <= value < MAX_VALUE:
                for other in range(0, 11):
                    pairs.add((value, other))
                    pairs.add((other, value))
        for digit in range(1, 10):
            run = digit * (power - 1) // 9
            if run < MAX_VALUE:
                pairs.add((run, 1))
                pairs.add((1, run))
                pairs.add((run, power - 1 - run))
    return sorted(pairs)


def boundary_examples(count):
    a_digits = torch.zeros((count, 14), dtype=torch.long, device=DEVICE)
    b_digits = torch.zeros((count, 14), dtype=torch.long, device=DEVICE)
    position = torch.randint(0, 14, (count,), device=DEVICE)
    left = torch.randint(1, 10, (count,), device=DEVICE)
    left[:count // 2] = 5
    rows = torch.arange(count, device=DEVICE)
    a_digits[rows, position] = left
    b_digits[rows, position] = 10 - left
    extra = torch.rand((count, 14), device=DEVICE) < 0.12
    extra[rows, position] = False
    values = torch.randint(0, 10, (count, 14), device=DEVICE)
    a_digits = torch.where(extra, values, a_digits)
    ceilings = 9 - a_digits
    other = (torch.rand((count, 14), device=DEVICE) * (ceilings + 1)).long()
    b_digits = torch.where(extra, other, b_digits)
    return from_digits(a_digits), from_digits(b_digits)


def calibration_batch(batch_size):
    a, b, targets = make_batch(batch_size, 0.5)
    count = batch_size // 8
    values_a, values_b = boundary_examples(count)
    sentinel = torch.full((count, 1), 10, dtype=torch.long, device=DEVICE)
    a[:count] = torch.cat((to_digits(values_a, 14), sentinel), dim=1)
    b[:count] = torch.cat((to_digits(values_b, 14), sentinel), dim=1)
    targets[:count] = to_digits(values_a + values_b, 15)
    return a, b, targets


def calibrate(model, steps=4000, batch_size=4096):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.001, fused=True)
    best_state = None
    best_errors = 10 ** 9
    pairs = systematic_cases()
    model.train()
    for step in range(steps):
        a, b, targets = calibration_batch(batch_size)
        optimizer.zero_grad(set_to_none=True)
        logits = model(a, b, targets[:, :-1])[:, 14:]
        loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 500 == 0:
            model.eval()
            edge_errors = evaluate_pairs(model, pairs)
            random_errors = evaluate(model, 2, 8192, "uniform")[0]
            score = len(edge_errors) * 100 + random_errors
            print("calibration", step + 1, loss.item(), "edge", len(edge_errors), "random", random_errors, flush=True)
            if score < best_errors:
                best_errors = score
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    export_model(model, None)


@torch.inference_mode()
def evaluate_pairs(model, pairs, batch_size=4096):
    errors = []
    for offset in range(0, len(pairs), batch_size):
        chunk = pairs[offset:offset + batch_size]
        a_values = torch.tensor([x[0] for x in chunk], device=DEVICE)
        b_values = torch.tensor([x[1] for x in chunk], device=DEVICE)
        targets = to_digits(a_values + b_values, 15)
        sentinel = torch.full((len(chunk), 1), 10, dtype=torch.long, device=DEVICE)
        a_tokens = torch.cat((to_digits(a_values, 14), sentinel), dim=1)
        b_tokens = torch.cat((to_digits(b_values, 14), sentinel), dim=1)
        generated = torch.empty((len(chunk), 0), dtype=torch.long, device=DEVICE)
        for _ in range(15):
            generated = torch.cat((generated, model(a_tokens, b_tokens, generated)[:, -1].argmax(-1, keepdim=True)), 1)
        bad = (generated != targets).any(1).nonzero().flatten().tolist()
        errors.extend(chunk[index] for index in bad)
    return errors


def export_model(model, module):
    state = {name: tensor.detach().float().cpu().tolist() for name, tensor in model.state_dict().items()}
    source_path = ROOT / "submission.py"
    source = source_path.read_text()
    prefix, marker, _ = source.partition("WEIGHTS = ")
    remainder = source[source.index("\n\n\nclass Block", len(prefix)):]
    source_path.write_text(prefix + "WEIGHTS = " + repr(state) + remainder)
    torch.save(model.state_dict(), ROOT / "checkpoint.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(26014)
    torch.set_float32_matmul_precision("high")
    module = load_architecture()
    model = module.AdditionTransformer().to(DEVICE)
    if args.resume and (ROOT / "checkpoint.pt").exists():
        model.load_state_dict(torch.load(ROOT / "checkpoint.pt", weights_only=True))
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.005, fused=True)
    milestones = ((0, 3e-3, 0.20), (8000, 1e-3, 0.25), (15000, 3e-4, 0.35), (21000, 1e-4, 0.45))
    best_score = math.inf
    for step in range(args.steps):
        lr, fraction = milestones[0][1:]
        for begin, candidate_lr, candidate_fraction in milestones:
            if step >= begin:
                lr, fraction = candidate_lr, candidate_fraction
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, args.batch_size, fraction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1} loss {loss.item():.7f} lr {lr:g}", flush=True)
        if (step + 1) % 2000 == 0 and step >= 7999:
            model.eval()
            uniform = evaluate(model, 2, 8192, "uniform")
            carry = evaluate(model, 1, 8192, "carry")
            noncarry = evaluate(model, 1, 8192, "noncarry")
            score = uniform[0] + carry[0] + noncarry[0]
            print("validation", uniform, carry, noncarry, flush=True)
            if score <= best_score:
                best_score = score
                export_model(model, module)
                print("saved score", score, flush=True)
            model.train()
    model.eval()
    for kind in ("uniform", "carry", "noncarry", "mixed"):
        result = evaluate(model, 8, 8192, kind)
        print("final", kind, result, flush=True)
    edge_errors = evaluate_pairs(model, systematic_cases())
    print("systematic", len(systematic_cases()), "errors", len(edge_errors), edge_errors[:20], flush=True)
    export_model(model, module)


if __name__ == "__main__":
    main()
