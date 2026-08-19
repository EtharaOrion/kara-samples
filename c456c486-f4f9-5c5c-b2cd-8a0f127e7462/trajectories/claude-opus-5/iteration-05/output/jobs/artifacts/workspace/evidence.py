"""Evidence that the attention is real and input-dependent, not a fixed pattern."""
import importlib.util, torch
spec = importlib.util.spec_from_file_location('sub', '/workspace/submission.py')
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
model, _ = sub.build_model()

def attn(a, b):
    oh = torch.zeros(1, 16, 10); oh[0, 0, 0] = 2.0
    for n in (a, b):
        for i in range(15):
            oh[0, i + 1, n % 10] += 1.0; n //= 10
    with torch.no_grad():
        z = oh @ model.U
        h = torch.relu(z.unsqueeze(-1) * model.W1a + model.b1a)
        c1 = (h * model.W2a0).sum(-1)
        sc = (c1 + model.bq).unsqueeze(-1) * c1.unsqueeze(-2) + model.slope * model.rel
        A = torch.softmax(sc + model.mA, -1)[0]
    return A

print('Attention source (argmax over keys) for slot i, on three inputs.')
print('Slot i attends to the nearest earlier slot whose digit sum != 9 -- the carry')
print('lookahead.  Slot 0 is the sentinel; slot k holds digit k-1.\n')
cases = [(11111111111111, 11111111111111), (55555555555555, 44444444444444),
         (12345678901234, 98765432109876)]
for a, b in cases:
    A = attn(a, b)
    ds = [((a // 10 ** i) % 10) + ((b // 10 ** i) % 10) for i in range(15)]
    print(f'a={a:014d} b={b:014d}')
    print('  digit sums (slots 1..15):', ds)
    print('  attends to slot:         ', A.argmax(-1).tolist()[1:])
    print('  max weight:              ', [round(float(x), 3) for x in A.max(-1).values[1:]])
    print()

A1, A2 = attn(*cases[0]), attn(*cases[1])
print(f'max |A(input1) - A(input2)| = {float((A1-A2).abs().max()):.4f}   '
      '(0 would mean a fixed pattern)')
print(f'rows whose argmax differs between the two inputs: '
      f'{int((A1.argmax(-1)!=A2.argmax(-1)).sum())}/16')

# ablation: freeze attention to the fixed "previous slot" pattern
print('\nAblation -- replace the learned attention with a fixed previous-slot pattern:')
import torch.nn.functional as F
orig = model.forward
def fixed_forward(oh):
    z = oh @ model.U
    h = torch.relu(z.unsqueeze(-1) * model.W1a + model.b1a)
    c2 = (h[..., 1:] * model.W2a1p).sum(-1)
    prev = torch.zeros(16, 16); prev[0, 0] = 1.0
    for i in range(1, 16): prev[i, i - 1] = 1.0
    c_in = c2 @ prev.T
    cur = torch.eye(16)
    c_out = c2 @ cur.T
    y = z + model.wo[0] * c_in + model.wo[1] * c_out
    V = model.U
    return (2.0 * y.unsqueeze(-1) * V - V * V)[:, 1:, :]
import random
rnd = random.Random(0); ok_l = ok_f = 0
for _ in range(2000):
    a = rnd.randint(0, 99999999999999); b = rnd.randint(0, 99999999999999)
    ok_l += sub.add(model, a, b) == a + b
    model.forward = fixed_forward
    ok_f += sub.add(model, a, b) == a + b
    model.forward = orig
print(f'  learned attention: {ok_l}/2000 correct')
print(f'  fixed  attention:  {ok_f}/2000 correct')
