import torch
from submission import Adder
from train import evaluate, POW10

DEVICE = "cuda"
model = Adder().to(DEVICE)
model.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
model.eval()

print("random", evaluate(model, batches=32, structured=0.0, batch=8192))
print("mixed structured", evaluate(model, batches=32, structured=0.85, batch=8192))

pairs = {(0, 0), (99_999_999_999_999, 99_999_999_999_999),
         (99_999_999_999_999, 1), (1, 99_999_999_999_999),
         (50_000_000_000_000, 50_000_000_000_000)}
for start in range(14):
    for length in range(1, 15 - start):
        run = ((10 ** length) - 1) * (10 ** start)
        pairs.add((run, 10 ** start))
        pairs.add((10 ** start, run))
        pairs.add((run, run))
        pairs.add((99_999_999_999_999 - run, run))
for d in range(10):
    x = int(str(d) * 14)
    for e in range(10):
        pairs.add((x, int(str(e) * 14)))
for k in range(14):
    pairs.add((10 ** k, 99_999_999_999_999 - 10 ** k))
    pairs.add((99_999_999_999_999 - 10 ** k, 10 ** k))

pairs = list(pairs)
a = torch.tensor([p[0] for p in pairs], device=DEVICE)
b = torch.tensor([p[1] for p in pairs], device=DEVICE)
p10 = POW10[:14]
da = (a[:, None] // p10) % 10
db = (b[:, None] // p10) % 10
target = ((a + b)[:, None] // POW10[:15]) % 10
prefix = torch.empty((len(pairs), 30), dtype=torch.long, device=DEVICE)
prefix[:, 0] = 10
prefix[:, 1:29:2] = da
prefix[:, 2:29:2] = db
prefix[:, 29] = 11
pred = []
with torch.inference_mode():
    for _ in range(15):
        digit = model(prefix)[:, -1].argmax(-1)
        pred.append(digit)
        prefix = torch.cat((prefix, digit[:, None]), 1)
pred = torch.stack(pred, 1)
bad = (pred != target).any(1)
print("systematic", int(bad.sum()), "/", len(pairs))
if bad.any():
    for i in bad.nonzero()[:20]:
        j = int(i)
        print(pairs[j], pred[j].tolist(), target[j].tolist())

# Explicitly inspect the first block's pre-softmax attention scores.
def scores(tokens):
    x = model.token(tokens) + model.position[:tokens.shape[1]]
    q, k, _ = model.blocks[0].qkv(model.blocks[0].ln1(x)).chunk(3, -1)
    q = q.view(tokens.shape[0], tokens.shape[1], 2, 8).transpose(1, 2)
    k = k.view(tokens.shape[0], tokens.shape[1], 2, 8).transpose(1, 2)
    return q @ k.transpose(-2, -1)
t1 = prefix[:1, :30].clone()
t2 = t1.clone(); t2[:, 1:29] = (t2[:, 1:29] + 3) % 10
print("qk max input change", float((scores(t1) - scores(t2)).abs().max()))
