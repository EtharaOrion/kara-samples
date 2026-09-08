"""Trainer for the baseline submission. NOT graded and NOT statically screened.

The graded file, `submission.py`, may not contain training code: the grader
screens it and rejects the imports and the operand arithmetic that generating
labelled data requires. So training lives here, and the weights it produces are
written back into `submission.py` as a literal.

Run it:

    python train.py                 # train with the shipped settings
    python train.py --steps 20000   # train longer

Improving this file is as much a part of the task as improving the model. The
shipped settings do not reach the accuracy threshold; see the docstring in
`submission.py` for where this baseline stands.
"""

import argparse
import random

import torch
import torch.nn.functional as F

import submission
from submission import (
    DIGITS, EOS, RESULT_DIGITS, VOCAB, Subtractor3Transformer, digits_of
)

# Index in the target sequence where the difference digits begin. Everything before
# it is operand tokens, which are random and therefore unpredictable; training
# on them spends capacity learning noise. IGNORE is the standard cross-entropy
# sentinel for "do not supervise this position".
ANSWER_START = 3 * DIGITS + 3
IGNORE = -100


def truth(a, b, c):
    """The quantity the graded file has to reproduce."""
    return a - b - c


def draw_operands(rng, limit):
    """One training tuple, drawn the way the grader draws its graded cases.

    Three independent uniforms over the full-width band, redrawn until
    `b + c <= a`. Rejection is the only draw that is uniform over the
    admitted domain; a deterministic reordering over-weights small
    subtrahends and under-represents the long double-borrow chains the
    task exists to test. Roughly one draw in nine is accepted.
    """
    floor = 10 ** (DIGITS - 1)
    while True:
        a = rng.randint(floor, limit)
        b = rng.randint(floor, limit)
        c = rng.randint(floor, limit)
        if b + c <= a:
            return a, b, c


def encode_with_answer(operands, total):
    tokens = submission.encode(*operands)
    tokens += digits_of(total, RESULT_DIGITS)
    tokens.append(EOS)
    return tokens


def batch(size, rng):
    limit = 10 ** DIGITS - 1
    xs, ys = [], []
    for _ in range(size):
        operands = draw_operands(rng, limit)
        seq = encode_with_answer(operands, truth(*operands))
        xs.append(seq[:-1])
        target = seq[1:]
        ys.append([IGNORE] * ANSWER_START + target[ANSWER_START:])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def train(steps=2000, batch_size=256, lr=3e-3, seed=0, log_every=100):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Subtractor3Transformer()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(steps, 1)
    )
    for step in range(steps):
        xs, ys = batch(batch_size, rng)
        logits = model(xs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), ys.reshape(-1), ignore_index=IGNORE
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if log_every and step % log_every == 0:
            print("step {} loss {:.4f}".format(step, float(loss.detach())))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, trials=200, seed=1234):
    """Exact-match accuracy, the same predicate the grader uses."""
    rng = random.Random(seed)
    limit = 10 ** DIGITS - 1
    correct = 0
    for _ in range(trials):
        operands = draw_operands(rng, limit)
        correct += int(
            submission.subtract3(model, *operands) == truth(*operands)
        )
    return correct / trials


def embed(model, path="submission.py"):
    """Write the trained weights into the graded file as a literal."""
    payload = {
        key: value.detach().cpu().tolist() for key, value in model.state_dict().items()
    }
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    marker = "EMBEDDED_WEIGHTS = "
    head = text[: text.rindex(marker) + len(marker)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + repr(payload) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    trained = train(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
    print("held-out exact-match accuracy:", evaluate(trained, trials=args.trials))
    if not args.no_embed:
        embed(trained)
        print("weights embedded into submission.py")
