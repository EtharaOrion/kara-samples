import importlib.util
import pathlib
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('/workspace')
spec = importlib.util.spec_from_file_location('trainer', ROOT / 'train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

def encode(a, b):
    size = a.shape[0]
    p8 = 10 ** torch.arange(8, device=a.device)
    ad, bd = (a[:, None] // p8) % 10, (b[:, None] // p8) % 10
    operands = torch.stack((ad, bd), 2).reshape(size, 16)
    p9 = 10 ** torch.arange(9, device=a.device)
    result = ((a + b)[:, None] // p9) % 10
    x = torch.empty((size, 25), dtype=torch.long, device=a.device)
    x[:, :16], x[:, 16], x[:, 17:] = operands, 10, result[:, :-1]
    return x, result

model = trainer.submission.AdditionTransformer(24, 4, 48, 2).cuda()
model.load_state_dict(torch.load(ROOT / 'candidate.pt', weights_only=True))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
g = torch.Generator(device='cuda').manual_seed(314159)
model.train()
for step in range(1, 3001):
    a = torch.randint(10_000_000, 90_000_001, (1024,), device='cuda', generator=g)
    b = 100_000_000 - a
    ar = torch.randint(10_000_000, 100_000_000, (3072,), device='cuda', generator=g)
    br = torch.randint(10_000_000, 100_000_000, (3072,), device='cuda', generator=g)
    x, target = encode(torch.cat((a, ar)), torch.cat((b, br)))
    logits = model(x)[:, 16:25]
    loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    if step % 500 == 0: print(step, loss.item(), flush=True)
torch.save(model.state_dict(), ROOT / 'candidate.pt')
