import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace")
from train import Adder, mixed_batch, result_digits, evaluate, export_submission, N

DEVICE = "cuda"
torch.manual_seed(2108)
model = Adder(10, 8).to(DEVICE)
model.load_state_dict(torch.load("/workspace/model_h8.pt", weights_only=True))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.001)


def long_carry_batch(batch):
    # Full-sequence additions whose incoming carry traverses an arbitrary run.
    a = torch.randint(0, 10, (batch, N), device=DEVICE)
    b = torch.randint(0, 10, (batch, N), device=DEVICE)
    a[:, -1] = 0
    b[:, -1] = 0
    rows = torch.arange(batch, device=DEVICE)
    start = torch.randint(0, 14, (batch,), device=DEVICE)
    # Bias toward reaching the most significant operand columns.
    use_top = torch.rand(batch, device=DEVICE) < 0.65
    end = torch.where(use_top, torch.full_like(start, 14), start + 1 + (torch.rand(batch, device=DEVICE) * (14-start)).long())
    pos = torch.arange(14, device=DEVICE)[None]
    inside = (pos > start[:, None]) & (pos < end[:, None])
    av = torch.randint(0, 10, (batch, 14), device=DEVICE)
    a[:, :14] = torch.where(inside, av, a[:, :14])
    b[:, :14] = torch.where(inside, 9-av, b[:, :14])
    ia = torch.randint(1, 10, (batch,), device=DEVICE)
    a[rows, start] = ia
    b[rows, start] = 10-ia + torch.randint(0, 10, (batch,), device=DEVICE).remainder(ia)
    stop = end < 14
    sr, sp = rows[stop], end[stop]
    sa = torch.randint(0, 9, (sr.numel(),), device=DEVICE)
    a[sr, sp] = sa
    b[sr, sp] = torch.randint(0, 9, (sr.numel(),), device=DEVICE).remainder(9-sa)
    return a, b, result_digits(a,b)

for step in range(1, 6001):
    a,b,y = mixed_batch(4096, 0.45)
    ca,cb,cy = long_carry_batch(2048)
    a[:2048],b[:2048],y[:2048] = ca,cb,cy
    logits=model(a,b)
    loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    if step % 1000 == 0:
        print(step, loss.item(), evaluate(model,16,8192,False), evaluate(model,8,8192,True), flush=True)

torch.save(model.state_dict(), "/workspace/model_h8_calibrated.pt")
export_submission(model, "/workspace/submission.py")
