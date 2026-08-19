import os, random, time
import torch
import torch.nn.functional as F
from model import Adder

DEVICE = 'cuda'
BATCH = int(os.environ.get('BATCH', '4096'))
STEPS = int(os.environ.get('STEPS', '24000'))
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def answers(a, b):
    out = torch.empty_like(a)
    carry = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    for j in range(15):
        total = a[:, j] + b[:, j] + carry
        out[:, j] = total.remainder(10)
        carry = total.div(10, rounding_mode='floor')
    return out

def batch_data(n, structured=True):
    a = torch.randint(0, 10, (n, 15), device=DEVICE)
    b = torch.randint(0, 10, (n, 15), device=DEVICE)
    a[:, 14] = 0; b[:, 14] = 0
    if structured:
        # Each batch regenerates varied full-operand patterns; no transition table is enumerated.
        q = n // 8
        col = torch.arange(14, device=DEVICE)[None]
        # True carry chains with random starts and lengths.
        lo, hi = 0, q
        start = torch.randint(0, 14, (q, 1), device=DEVICE)
        length = torch.randint(1, 15, (q, 1), device=DEVICE)
        end = torch.minimum(start + length, torch.tensor(14, device=DEVICE))
        run = (col >= start) & (col < end)
        first = col == start
        av = torch.randint(1, 10, (q, 14), device=DEVICE)
        bv = 9 - av
        bv = torch.where(first, 10 - av, bv)
        a[lo:hi, :14] = torch.where(run, av, a[lo:hi, :14])
        b[lo:hi, :14] = torch.where(run, bv, b[lo:hi, :14])
        # Matched non-carry 9-runs, important to prevent hallucinated carries.
        lo, hi = q, 2*q
        start = torch.randint(0, 14, (q, 1), device=DEVICE)
        length = torch.randint(1, 15, (q, 1), device=DEVICE)
        run = (col >= start) & (col < torch.minimum(start + length, torch.tensor(14, device=DEVICE)))
        av = torch.randint(0, 10, (q, 14), device=DEVICE)
        a[lo:hi, :14] = torch.where(run, av, a[lo:hi, :14])
        b[lo:hi, :14] = torch.where(run, 9-av, b[lo:hi, :14])
        # Sparse increments into random 9-runs.
        lo, hi = 2*q, 3*q
        a[lo:hi] = 0; b[lo:hi] = 0
        start = torch.randint(0, 14, (q, 1), device=DEVICE)
        length = torch.randint(1, 15, (q, 1), device=DEVICE)
        run = (col >= start) & (col < torch.minimum(start + length, torch.tensor(14, device=DEVICE)))
        a[lo:hi, :14] = run.long() * 9
        b[lo:hi, :14] = (col == start).long()
        # Repeated and blockwise operands.
        lo, hi = 3*q, 4*q
        da = torch.randint(0, 10, (q, 1), device=DEVICE)
        db = torch.randint(0, 10, (q, 1), device=DEVICE)
        a[lo:hi, :14] = da; b[lo:hi, :14] = db
        # Complementary columns with occasional initiating carries.
        lo, hi = 4*q, 5*q
        av = torch.randint(0, 10, (q, 14), device=DEVICE)
        a[lo:hi, :14] = av; b[lo:hi, :14] = 9-av
        bump = torch.randint(0, 14, (q,), device=DEVICE)
        rows = torch.arange(q, device=DEVICE)
        can = a[lo:hi, :14][rows, bump] > 0
        b[lo:hi, :14][rows[can], bump[can]] += 1
        # Small and sparse general operands.
        lo, hi = 5*q, 6*q
        keep = torch.rand((q, 14), device=DEVICE) < .18
        a[lo:hi, :14] *= keep; b[lo:hi, :14] *= (torch.rand((q, 14), device=DEVICE) < .18)
        # Top-boundary overflow chains.
        lo, hi = 6*q, 7*q
        length = torch.randint(1, 15, (q, 1), device=DEVICE)
        run = col >= (14-length)
        av = torch.randint(0, 10, (q, 14), device=DEVICE)
        a[lo:hi, :14] = torch.where(run, av, a[lo:hi, :14])
        b[lo:hi, :14] = torch.where(run, 9-av, b[lo:hi, :14])
        s = (14-length).squeeze(1)
        rows = torch.arange(q, device=DEVICE)
        ok = a[lo:hi, :14][rows,s] > 0
        b[lo:hi, :14][rows[ok],s[ok]] += 1
    return a, b, answers(a, b)

def evaluate(model, batches=16, n=4096, structured=False):
    errors = 0
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            a,b,y = batch_data(n, structured)
            prev = torch.empty((n,0), dtype=torch.long, device=DEVICE)
            for j in range(15):
                digit = model(a,b,prev)[:,-1].argmax(-1,keepdim=True)
                prev = torch.cat((prev,digit),1)
            errors += (prev != y).any(1).sum().item()
    model.train()
    return errors, batches*n

def main():
    torch.manual_seed(2501); random.seed(2501)
    model = Adder().to(DEVICE)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(.9,.98), weight_decay=.003, fused=True)
    best = None; started=time.time()
    for step in range(1, STEPS+1):
        if step == 8001:
            for g in optimizer.param_groups: g['lr']=1e-3
        if step == 15001:
            for g in optimizer.param_groups: g['lr']=3e-4
        if step == 20001:
            for g in optimizer.param_groups: g['lr']=1e-4
        a,b,y = batch_data(BATCH, structured=True)
        logits = model(a,b,y[:,:-1])
        loss = F.cross_entropy(logits.reshape(-1,10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0 or step == 1:
            print(step, float(loss), optimizer.param_groups[0]['lr'], 'seconds', int(time.time()-started), flush=True)
        if step >= 12000 and step % 2000 == 0:
            e1,n1=evaluate(model,8,4096,False); e2,n2=evaluate(model,8,4096,True)
            print('validation', e1,n1,e2,n2,flush=True)
            score=e1+e2
            if best is None or score <= best:
                best=score; torch.save(model.state_dict(), '/workspace/model.pt')
    torch.save(model.state_dict(), '/workspace/model_final.pt')

if __name__ == '__main__': main()
