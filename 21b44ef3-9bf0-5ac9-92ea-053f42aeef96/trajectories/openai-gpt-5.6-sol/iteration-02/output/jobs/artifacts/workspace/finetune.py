import importlib.util
import pathlib
import torch
import torch.nn.functional as F

ROOT = pathlib.Path('/workspace')
spec = importlib.util.spec_from_file_location('trainer', ROOT / 'train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

model = trainer.submission.AdditionTransformer(24, 4, 48, 2).cuda()
model.load_state_dict(torch.load(ROOT / 'candidate.pt', weights_only=True))
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.001)
generator = torch.Generator(device='cuda').manual_seed(7654321)
model.train()
for step in range(1, 5001):
    x, target = trainer.batch(4096, torch.device('cuda'), generator)
    logits = model(x)[:, 16:25]
    loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 500 == 0:
        exact, digit = trainer.validate(model, batches=10, seed=123000 + step)
        print(f'step={step} loss={loss.item():.6f} exact={exact:.6%} digit={digit:.6%}', flush=True)
exact, digit = trainer.validate(model, batches=100, seed=24681357)
print(f'final exact={exact:.6%} digit={digit:.6%}', flush=True)
torch.save(model.state_dict(), ROOT / 'candidate.pt')
