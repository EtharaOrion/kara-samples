"""Grader-style check of /workspace/submission.py (never imported by it)."""
import importlib.util, random, sys, time
import torch

spec = importlib.util.spec_from_file_location('submission', sys.argv[1] if len(sys.argv) > 1
                                              else '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

model, meta = sub.build_model()
n = sum(p.numel() for p in model.parameters())
print('metadata:', meta)
print('registered parameters:', n)
assert isinstance(model, torch.nn.Module)

MAX = 10 ** 14 - 1
rng = random.Random(12345)

edge = [(0, 0), (0, MAX), (MAX, MAX), (MAX, 1), (1, MAX), (99999999999999, 99999999999999),
        (50000000000000, 50000000000000), (9999999, 1), (999999999999, 1), (10 ** 13, 10 ** 13),
        (123456789, 987654321), (5, 5), (0, 1), (1, 0), (10 ** 14 - 2, 1), (12345678901234, 0),
        (99999999999998, 1), (55555555555555, 44444444444445), (11111111111111, 88888888888889),
        (7, 3)]
bad = [(a, b) for a, b in edge if sub.add(model, a, b) != a + b]
print(f'edge cases: {len(edge)-len(bad)}/{len(edge)}', 'FAIL' + str(bad[:5]) if bad else 'ok')

t0 = time.time()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
wrong = []
for _ in range(N):
    a = rng.randint(0, MAX); b = rng.randint(0, MAX)
    if sub.add(model, a, b) != a + b:
        wrong.append((a, b))
print(f'uniform: {(N-len(wrong))/N:.5f} on {N} pairs ({time.time()-t0:.0f}s), wrong={wrong[:5]}')

# mixed-magnitude / carry-heavy pairs
wrong2 = []
for _ in range(N):
    la, lb = rng.randint(1, 14), rng.randint(1, 14)
    a = rng.randint(0, 10 ** la - 1); b = rng.randint(0, 10 ** lb - 1)
    if rng.random() < 0.3:
        b = int(''.join(str(9 - int(c)) for c in str(a)))
    if sub.add(model, a, b) != a + b:
        wrong2.append((a, b))
print(f'stress:  {(N-len(wrong2))/N:.5f}, wrong={wrong2[:5]}')
