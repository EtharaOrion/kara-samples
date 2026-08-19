import importlib.util
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
from submission import Adder

DEVICE = 'cuda'
LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10 ** i for i in range(15)], device=DEVICE, dtype=torch.long)

def make_values(batch, structured=0.0):
    a = torch.randint(0, LIMIT, (batch,), device=DEVICE)
    b = torch.randint(0, LIMIT, (batch,), device=DEVICE)
    if structured <= 0:
        return a, b
    use = torch.rand(batch, device=DEVICE) < structured
    idx = use.nonzero().flatten()
    if not idx.numel(): return a, b
    n = idx.numel()
    kind = torch.randint(0, 7, (n,), device=DEVICE)
    pos = torch.randint(0, 14, (n,), device=DEVICE)
    length = torch.minimum(torch.randint(1, 15, (n,), device=DEVICE), 14-pos)
    scale = POW10[pos]
    span = POW10[length]
    prefix_cap = torch.div(torch.full_like(span, LIMIT-1), scale*span, rounding_mode='floor') + 1
    prefix = torch.floor(torch.rand(n, device=DEVICE) * prefix_cap).long()
    low = torch.floor(torch.rand(n, device=DEVICE) * scale).long()
    run9 = prefix*scale*span + (span-1)*scale + low
    run8 = prefix*scale*span + torch.div(8*(span-1), 9, rounding_mode='floor')*scale + low
    x = torch.randint(0, LIMIT, (n,), device=DEVICE)
    y = torch.randint(0, LIMIT, (n,), device=DEVICE)
    # long carries; matched near non-carries
    x = torch.where(kind == 0, run9, x); y = torch.where(kind == 0, scale, y)
    x = torch.where(kind == 1, run8, x); y = torch.where(kind == 1, scale, y)
    # isolated 5+5 and 9+9 columns
    d = torch.where(torch.rand(n, device=DEVICE) < .5, 5, 9)
    x = torch.where(kind == 2, d*scale, x); y = torch.where(kind == 2, d*scale, y)
    # sparse random digit pairs at arbitrary positions
    da = torch.randint(0, 10, (n,), device=DEVICE); db = torch.randint(0, 10, (n,), device=DEVICE)
    x = torch.where(kind == 3, da*scale, x); y = torch.where(kind == 3, db*scale, y)
    # complements around powers of ten
    cap = torch.minimum(POW10[torch.clamp(pos+length, max=14)], torch.full_like(scale, LIMIT))
    lo = torch.minimum(torch.randint(1, 1000, (n,), device=DEVICE), cap)
    x = torch.where(kind == 4, cap-lo, x); y = torch.where(kind == 4, lo, y)
    # repeated digits
    repa = torch.randint(0, 10, (n,), device=DEVICE) * 11111111111111
    repb = torch.randint(0, 10, (n,), device=DEVICE) * 11111111111111
    x = torch.where(kind == 5, repa, x); y = torch.where(kind == 5, repb, y)
    # shifted exact all-nine run plus one
    x = torch.where(kind == 6, (span-1)*scale, x); y = torch.where(kind == 6, scale, y)
    a[idx] = x; b[idx] = y
    return a, b

def digitize(v, count):
    return torch.div(v[:, None], POW10[:count], rounding_mode='floor').remainder(10)

def batch_data(batch, structured):
    a, b = make_values(batch, structured)
    ad = digitize(a, 14)
    bd = digitize(b, 14)
    sentinel = torch.full((batch,1), 10, device=DEVICE, dtype=torch.long)
    target = digitize(a+b, 15)
    return torch.cat((ad,sentinel),1), torch.cat((bd,sentinel),1), target

def loss_for(model, batch, structured):
    ad, bd, target = batch_data(batch, structured)
    logits = model(ad, bd, target[:,:-1])[:,14:]
    return F.cross_entropy(logits.reshape(-1,10), target.reshape(-1))

@torch.inference_mode()
def evaluate(model, total=131072, structured=0.0, batch=4096):
    errors=0
    for _ in range((total+batch-1)//batch):
        n=min(batch,total)
        ad,bd,target=batch_data(n,structured)
        prev=torch.empty((n,0),dtype=torch.long,device=DEVICE)
        for k in range(15):
            pred=model(ad,bd,prev)[:,-1].argmax(-1)
            prev=torch.cat((prev,pred[:,None]),1)
        errors += (prev != target).any(1).sum().item(); total-=n
    return errors

def export(model):
    import submission
    flat=torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float().numpy().tobytes().hex()
    path=Path('/workspace/submission.py')
    text=path.read_text()
    left=text.index("_WEIGHTS = '")+len("_WEIGHTS = '")
    right=text.index("'",left)
    path.write_text(text[:left]+flat+text[right:])

if __name__ == '__main__':
    torch.manual_seed(40)
    model=Adder().to(DEVICE)
    print('parameters',sum(p.numel() for p in model.parameters()),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,betas=(.9,.98),weight_decay=.003)
    phases=[(14000,3e-3,0.0),(14000,1e-3,.35),(14000,3e-4,.48),(14000,1e-4,.48),(10000,3e-5,.45),(6000,1e-5,.35)]
    step=0; start=time.time()
    for steps,lr,mix in phases:
        for g in opt.param_groups:g['lr']=lr
        for j in range(steps):
            opt.zero_grad(set_to_none=True)
            loss=loss_for(model,4096,mix)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            step+=1
            if step%2000==0: print(step,float(loss),time.time()-start,flush=True)
        torch.save(model.state_dict(),'/workspace/checkpoint.pt')
    model.eval()
    print('uniform errors',evaluate(model,262144,0),flush=True)
    print('mixed errors',evaluate(model,262144,.7),flush=True)
    export(model)
