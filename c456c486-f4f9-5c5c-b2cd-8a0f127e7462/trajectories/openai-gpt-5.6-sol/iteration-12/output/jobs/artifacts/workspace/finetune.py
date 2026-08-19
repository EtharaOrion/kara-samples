import math
import time
import torch
import torch.nn.functional as F
from train import AdditionTransformer, batch_random, targets, evaluate


def long_carries(n, device):
    a = torch.randint(0, 10, (n, 15), device=device)
    b = torch.randint(0, 10, (n, 15), device=device)
    a[:, 14] = b[:, 14] = 0
    rows = torch.arange(n, device=device)
    # Half begin at the least-significant digit; all are deliberately long.
    starts = torch.where(rows.remainder(2) == 0, 0, torch.randint(0, 5, (n,), device=device))
    min_len = torch.minimum(torch.full_like(starts, 9), 14 - starts)
    lengths = min_len + (torch.rand(n, device=device) * (15 - starts - min_len)).long()
    # A quarter propagates exactly through the highest operand position.
    lengths = torch.where(rows.remainder(4) == 0, 14 - starts, lengths)
    for p in range(14):
        first = starts == p
        inside = (starts < p) & (p < starts + lengths)
        ending = starts + lengths == p
        x = torch.randint(1, 10, (n,), device=device)
        extra = torch.randint(0, 10, (n,), device=device)
        extra = torch.minimum(extra, x - 1)
        a[:, p] = torch.where(first, x, a[:, p])
        b[:, p] = torch.where(first, 10 - x + extra, b[:, p])
        x = torch.randint(0, 10, (n,), device=device)
        a[:, p] = torch.where(inside, x, a[:, p])
        b[:, p] = torch.where(inside, 9 - x, b[:, p])
        x = torch.randint(0, 9, (n,), device=device)
        y = torch.randint(0, 9, (n,), device=device)
        y = torch.minimum(y, 8 - x)
        a[:, p] = torch.where(ending, x, a[:, p])
        b[:, p] = torch.where(ending, y, b[:, p])
    # Explicit sparse increment through a variable run of nines.
    sparse = rows.remainder(3) == 0
    a[sparse, :14] = 0
    b[sparse, :14] = 0
    run = torch.randint(1, 15, (n,), device=device)
    for p in range(14):
        use = sparse & (p < run)
        a[:, p] = torch.where(use, torch.full_like(rows, 9), a[:, p])
    b[sparse, 0] = 1
    swap = torch.rand(n, device=device) < 0.5
    old = a.clone()
    a = torch.where(swap[:, None], b, a)
    b = torch.where(swap[:, None], old, b)
    return a, b, targets(a, b)


model = AdditionTransformer().cuda()
model.load_state_dict(torch.load('/workspace/model.pt', weights_only=True))
model.train()
torch.set_float32_matmul_precision('high')
compiled = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
start=time.time()
for step in range(1, 4001):
    ar,br,yr=batch_random(2048,'cuda')
    aa,ba,ya=long_carries(1024,'cuda')
    from train import batch_structured
    ast,bst,yst=batch_structured(1024,'cuda')
    a=torch.cat((ar,aa,ast)); b=torch.cat((br,ba,bst)); y=torch.cat((yr,ya,yst))
    opt.zero_grad(set_to_none=True)
    loss=F.cross_entropy(compiled(a,b).flatten(0,1),y.flatten())
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step % 500 == 0:
        r,_=evaluate(model,8,8192)
        model.eval()
        with torch.no_grad():
            xa,xb,xy=long_carries(65536,'cuda'); e=(model(xa,xb).argmax(-1)==xy).all(1).float().mean().item()
        model.train()
        print(step,loss.item(),r,e,time.time()-start,flush=True)
torch.save(model.state_dict(),'/workspace/model.pt')
