import copy
import math
import random
import time
import torch
import torch.nn.functional as F
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 4096
POWERS = torch.tensor([10**i for i in range(15)], device=DEVICE, dtype=torch.long)
LIMIT = 100_000_000_000_000


def to_digits(x):
    return (x[:, None] // POWERS[None, :]) % 10


def uniform(n):
    a = torch.randint(0, LIMIT, (n,), device=DEVICE)
    b = torch.randint(0, LIMIT, (n,), device=DEVICE)
    return a, b


def carry_examples(n):
    # Random digits with a forced carry followed by a randomized run of complementary columns.
    ad = torch.randint(0, 10, (n, 15), device=DEVICE)
    bd = torch.randint(0, 10, (n, 15), device=DEVICE)
    ad[:, 14] = 0
    bd[:, 14] = 0
    start = torch.randint(0, 14, (n,), device=DEVICE)
    length = torch.randint(1, 15, (n,), device=DEVICE)
    rows = torch.arange(n, device=DEVICE)
    x = torch.randint(1, 10, (n,), device=DEVICE)
    ad[rows, start] = x
    bd[rows, start] = 10 - x
    cols = torch.arange(15, device=DEVICE)[None, :]
    chain = (cols > start[:, None]) & (cols <= torch.minimum(start + length, torch.full_like(start, 13))[:, None])
    left = torch.randint(0, 10, (n, 15), device=DEVICE)
    ad = torch.where(chain, left, ad)
    bd = torch.where(chain, 9 - left, bd)
    a = (ad * POWERS).sum(1)
    b = (bd * POWERS).sum(1)
    return a, b


def boundary_examples(n):
    kind = torch.randint(0, 5, (n,), device=DEVICE)
    k = torch.randint(0, 14, (n,), device=DEVICE)
    p = POWERS[k]
    run = POWERS[k + 1] - 1
    a = torch.randint(0, LIMIT, (n,), device=DEVICE)
    b = torch.randint(0, LIMIT, (n,), device=DEVICE)
    a = torch.where(kind == 0, run, a); b = torch.where(kind == 0, torch.ones_like(b), b)
    a = torch.where(kind == 1, LIMIT - 1, a); b = torch.where(kind == 1, p, b)
    a = torch.where(kind == 2, 5 * p, a); b = torch.where(kind == 2, 5 * p, b)
    a = torch.where(kind == 3, 9 * p, a); b = torch.where(kind == 3, 9 * p, b)
    rep = torch.randint(0, 10, (n,), device=DEVICE) * 11_111_111_111_111
    a = torch.where(kind == 4, rep, a); b = torch.where(kind == 4, torch.randint(0, 10, (n,), device=DEVICE) * 11_111_111_111_111, b)
    # max + power may exceed operand range only in sum, which is valid.
    return a, b


def batch(step):
    if step < 12000:
        a, b = uniform(BATCH)
    else:
        nu = BATCH // 2
        nc = BATCH // 3
        a0, b0 = uniform(nu)
        a1, b1 = carry_examples(nc)
        a2, b2 = boundary_examples(BATCH - nu - nc)
        a = torch.cat((a0, a1, a2)); b = torch.cat((b0, b1, b2))
        order = torch.randperm(BATCH, device=DEVICE)
        a, b = a[order], b[order]
    return to_digits(a), to_digits(b), to_digits(a + b)


@torch.no_grad()
def evaluate(model, count=131072, structured=False):
    model.eval()
    errors = 0
    for _ in range((count + BATCH - 1) // BATCH):
        if structured:
            a1,b1=carry_examples(BATCH//2); a2,b2=boundary_examples(BATCH-BATCH//2)
            a=torch.cat((a1,a2)); b=torch.cat((b1,b2))
        else: a,b=uniform(BATCH)
        pred=model(to_digits(a),to_digits(b)).argmax(-1)
        errors += (pred.ne(to_digits(a+b)).any(1)).sum().item()
    model.train()
    return errors


def main():
    torch.manual_seed(46); random.seed(46)
    model = AdditionTransformer().to(DEVICE)
    if __import__('os').path.exists('/workspace/latest.pt'):
        model.load_state_dict(torch.load('/workspace/latest.pt', weights_only=True)['model'])
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.002)
    best = None; best_score = 10**9
    phases = [(8000,3e-3),(18000,1e-3),(30000,3e-4),(42000,1e-4),(50000,3e-5)]
    start=time.time()
    for step in range(50000):
        for end,lr in phases:
            if step < end: break
        for g in opt.param_groups: g['lr']=lr
        da,db,target=batch(step)
        logits=model(da,db)
        loss=F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (step+1)%2000==0:
            er=evaluate(model,65536,False); es=evaluate(model,32768,True); score=er+2*es
            print(step+1, f"loss={loss.item():.6g}", "random",er,"structured",es,"sec",round(time.time()-start),flush=True)
            torch.save({'model':model.state_dict(),'step':step+1,'random_errors':er,'structured_errors':es},'/workspace/latest.pt')
            if score <= best_score:
                best_score=score; best=copy.deepcopy(model.state_dict()); torch.save({'model':best,'step':step+1},'/workspace/best.pt')
    model.load_state_dict(best)
    print('FINAL',evaluate(model,1048576,False),evaluate(model,524288,True),flush=True)

if __name__=='__main__': main()
