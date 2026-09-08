from importlib.machinery import SourceFileLoader
import os
import torch
import torch.nn.functional as F

sub = SourceFileLoader("sub", "/workspace/submission.py").load_module()
trainer = SourceFileLoader("trainer", "/workspace/train.py").load_module()
model, _ = sub.build_model()
model.cuda().train()
torch.save(model.state_dict(), "/workspace/pre_edge_state.pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.98), weight_decay=0.0)
p10 = trainer.POW10

vals = {10_000_000, 99_999_999}
for lead in range(1, 10):
    for run in range(1, 8):
        scale = 10 ** run
        for delta in (-2, -1, 0, 1, 2):
            for prefix in (0, 1, 9, 10, 11, 19, 20, 49, 50, 79, 80, 89, 90, 98, 99):
                value = lead * 10_000_000 + (prefix * 100_000 + scale - 1 + delta) % 10_000_000
                if 10_000_000 <= value <= 99_999_999:
                    vals.add(value)
vals = torch.tensor(sorted(vals), device="cuda")

for step in range(1, 6001):
    tokens, target = trainer.make_batch(trainer.BATCH, 0.35)
    n = trainer.BATCH // 2
    a = vals[torch.randint(0, len(vals), (n,), device="cuda")]
    b = vals[torch.randint(0, len(vals), (n,), device="cuda")]
    ad = (a[:, None] // p10[:8]) % 10
    bd = (b[:, None] // p10[:8]) % 10
    out = (a[:, None] + b[:, None]) // p10[:9] % 10
    tokens[:n, 0:16:2] = ad
    tokens[:n, 1:16:2] = bd
    tokens[:n, 16] = 10
    tokens[:n, 17:] = out[:, :8]
    target[:n] = out
    logits = model(tokens)
    loss = F.cross_entropy(logits[:, 16:].reshape(-1, 10), target.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 1000 == 0:
        print(step, float(loss), trainer.evaluate(model, 4, 0.0), trainer.evaluate(model, 4, 0.55), flush=True)
trainer.export(model)
