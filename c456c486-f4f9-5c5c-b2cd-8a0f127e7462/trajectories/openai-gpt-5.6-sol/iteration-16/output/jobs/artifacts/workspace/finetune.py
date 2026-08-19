import torch
import torch.nn.functional as F
import train

train.torch.set_float32_matmul_precision("high")
model = train.Model().to(train.DEVICE)
model.load_state_dict(torch.load("/workspace/model.pt", weights_only=True))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.001)
for step in range(1, 3001):
    a, b = train.random_batch(train.BATCH, .25)
    count = 768
    k = torch.randint(1, 15, (count,), device=train.DEVICE)
    p = 10 ** k
    family = torch.arange(count, device=train.DEVICE) % 4
    repeated = (p - 1) // 9
    aa = torch.where(family == 0, p - 1, torch.where(family == 1, p - 1, torch.where(family == 2, repeated * torch.randint(1, 10, (count,), device=train.DEVICE), train.MAXIMUM)))
    bb = torch.where(family == 0, p - 1, torch.where(family == 1, torch.ones_like(p), torch.where(family == 2, repeated * torch.randint(1, 10, (count,), device=train.DEVICE), p - 1)))
    a[:count], b[:count] = aa, bb
    sequence = train.encode(a, b)
    logits = model(sequence[:, :-1])[:, 29:44]
    loss = F.cross_entropy(logits.reshape(-1, 10), sequence[:, 30:].reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 500 == 0:
        random_accuracy = train.evaluate(model, 65536, 0)[0]
        print(step, loss.item(), random_accuracy, flush=True)
torch.save(model.state_dict(), "/workspace/model.pt")
print("final", train.evaluate(model, 524288, 0)[0], train.evaluate(model, 262144, .8)[0], flush=True)
