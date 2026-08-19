import copy
import importlib.util
import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as NF

D, H, FW = 9, 3, 6
LIMIT = 100_000_000_000_000
POW10 = torch.tensor([10**i for i in range(15)], dtype=torch.long)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.n2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FW)
        self.fc2 = nn.Linear(FW, D)
    def forward(self, x):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        sh = (x.shape[0], x.shape[1], H, D // H)
        q, k, v = [t.view(sh).transpose(1, 2) for t in (q, k, v)]
        z = NF.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(z.transpose(1, 2).reshape_as(x))
        return x + self.fc2(NF.gelu(self.fc1(self.n2(x))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_embed = nn.Embedding(11, D)
        self.b_embed = nn.Embedding(11, D)
        self.out_embed = nn.Embedding(11, D)
        self.position = nn.Parameter(torch.empty(30, D))
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10)
        nn.init.normal_(self.position, std=.02)
    def forward(self, ad, bd, prev):
        n = ad.shape[0]
        src = self.a_embed(ad) + self.b_embed(bd) + self.out_embed.weight[10]
        absent = torch.full((n, prev.shape[1]), 10, dtype=torch.long, device=ad.device)
        dec = self.a_embed(absent) + self.b_embed(absent) + self.out_embed(prev)
        x = torch.cat((src, dec), 1) + self.position[:15 + prev.shape[1]]
        for block in self.blocks: x = block(x)
        return self.head(self.norm(x[:, 15:]))


def integer_digits(values):
    return torch.stack([(values // (10**i)) % 10 for i in range(15)], 1)


def labels(ad, bd):
    powers = POW10.to(ad.device)
    a = (ad * powers).sum(1)
    b = (bd * powers).sum(1)
    return integer_digits(a + b)


def low_sum_digits(n, device):
    total = torch.randint(0, 9, (n, 15), device=device)
    a = (torch.rand((n, 15), device=device) * (total + 1)).long()
    return a, total - a


def chains(n, device, carry=True):
    a, b = low_sum_digits(n, device)
    a[:, 14] = b[:, 14] = 0
    start = torch.randint(0, 14, (n,), device=device)
    length = torch.randint(1, 15, (n,), device=device)
    end = torch.minimum(start + length - 1, torch.full_like(start, 13))
    for j in range(14):
        if carry:
            initiate = start == j
            da = torch.randint(1, 10, (n,), device=device)
            a[:, j] = torch.where(initiate, da, a[:, j])
            b[:, j] = torch.where(initiate, 10 - da, b[:, j])
            propagate = (start < j) & (end >= j)
            da = torch.randint(0, 10, (n,), device=device)
            a[:, j] = torch.where(propagate, da, a[:, j])
            b[:, j] = torch.where(propagate, 9 - da, b[:, j])
        else:
            run = (start <= j) & (end >= j)
            da = torch.randint(0, 10, (n,), device=device)
            a[:, j] = torch.where(run, da, a[:, j])
            b[:, j] = torch.where(run, 9 - da, b[:, j])
    return a, b


def sparse(n, device):
    a = torch.zeros((n, 15), dtype=torch.long, device=device)
    b = torch.zeros_like(a)
    p = torch.randint(0, 14, (n,), device=device)
    rows = torch.arange(n, device=device)
    kind = torch.randint(0, 5, (n,), device=device)
    da = torch.where(kind == 0, 5, torch.randint(1, 10, (n,), device=device))
    totals = torch.where(kind <= 1, 10, torch.where(kind == 2, 9, torch.where(kind == 3, 11, 8)))
    da = torch.minimum(da, totals)
    db = totals - da
    bad = db > 9
    da = torch.where(bad, totals - 9, da)
    db = totals - da
    a[rows, p] = da
    b[rows, p] = db
    # Add occasional lower digits without creating an incoming carry.
    lower = torch.clamp(p - 1, min=0)
    use = (p > 0) & (kind == 4)
    a[rows[use], lower[use]] = 4
    b[rows[use], lower[use]] = 4
    return a, b


def repeated(n, device):
    da = torch.randint(0, 10, (n, 1), device=device)
    db = torch.randint(0, 10, (n, 1), device=device)
    a = da.expand(n, 15).clone(); b = db.expand(n, 15).clone()
    a[:, 14] = b[:, 14] = 0
    # Half use alternating/block patterns rather than only repeated operands.
    rows = torch.arange(n, device=device)
    alt = rows % 2 == 0
    second_a = torch.randint(0, 10, (n, 1), device=device)
    second_b = torch.randint(0, 10, (n, 1), device=device)
    for j in range(0, 14, 2):
        a[alt, j] = second_a[alt, 0]
        b[alt, j] = second_b[alt, 0]
    return a, b


def batch_data(batch, device, phase):
    # Every sample is a complete operand pair; only the mixture evolves.
    if phase == 0: sizes = [batch * 3 // 4, batch // 8, batch // 16]
    else: sizes = [batch // 2, batch // 4, batch // 8]
    sizes.append(batch - sum(sizes))
    vals = torch.randint(0, LIMIT, (sizes[0], 2), device=device)
    parts_a = [integer_digits(vals[:, 0])]
    parts_b = [integer_digits(vals[:, 1])]
    ca, cb = chains(sizes[1], device, True); parts_a.append(ca); parts_b.append(cb)
    if phase == 0: ca, cb = chains(sizes[2], device, False)
    else: ca, cb = sparse(sizes[2], device)
    parts_a.append(ca); parts_b.append(cb)
    if phase < 2: ca, cb = repeated(sizes[3], device)
    else: ca, cb = sparse(sizes[3], device)
    parts_a.append(ca); parts_b.append(cb)
    ad, bd = torch.cat(parts_a), torch.cat(parts_b)
    order = torch.randperm(batch, device=device)
    ad, bd = ad[order], bd[order]
    target = labels(ad, bd)
    prev = torch.cat((torch.full((batch, 1), 10, device=device, dtype=torch.long), target[:, :-1]), 1)
    return ad, bd, prev, target


@torch.inference_mode()
def ar_errors(model, n, device, kind='uniform', chunk=8192):
    errors = 0
    for begin in range(0, n, chunk):
        m = min(chunk, n - begin)
        if kind == 'uniform':
            vals = torch.randint(0, LIMIT, (m, 2), device=device)
            ad, bd = integer_digits(vals[:, 0]), integer_digits(vals[:, 1])
        elif kind == 'carry': ad, bd = chains(m, device, True)
        elif kind == 'noncarry': ad, bd = chains(m, device, False)
        elif kind == 'sparse': ad, bd = sparse(m, device)
        else: ad, bd = repeated(m, device)
        target = labels(ad, bd)
        prev = torch.full((m, 1), 10, dtype=torch.long, device=device)
        out = []
        for _ in range(15):
            d = model(ad, bd, prev)[:, -1].argmax(1)
            out.append(d); prev = torch.cat((prev, d[:, None]), 1)
        pred = torch.stack(out, 1)
        errors += (pred != target).any(1).sum().item()
    return errors


def export(model, path):
    state = {k: v.detach().cpu().float().tolist() for k, v in model.state_dict().items()}
    template = Path('/workspace/submission.py').read_text()
    marker = '\n\ndef build_model():'
    before, after = template.split(marker, 1)
    replacement = '''\n\n_WEIGHTS = __STATE__\n\ndef build_model():\n    model = AdditionTransformer()\n    model.load_state_dict({k: torch.tensor(v) for k, v in _WEIGHTS.items()})\n    model.eval()\n    return model, {"architecture": "aligned autoregressive transformer", "digits": 14}\n'''.replace('__STATE__', repr(state))
    # Keep add(), discard the old random build_model body.
    add_part = after.split('\n\ndef add(', 1)[1]
    Path(path).write_text(before + replacement + '\ndef add(' + add_part)


def main():
    torch.manual_seed(3109); random.seed(3109)
    device = torch.device('cuda')
    model = Model().to(device)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.003, fused=True)
    batch = 4096
    phases = [(9000, 3e-3, 0), (9000, 1e-3, 1), (8000, 3e-4, 1), (6000, 1e-4, 2)]
    step = 0; best = None; best_score = 10**9
    for count, lr, phase in phases:
        for group in opt.param_groups: group['lr'] = lr
        for _ in range(count):
            step += 1
            ad, bd, prev, target = batch_data(batch, device, phase)
            opt.zero_grad(set_to_none=True)
            logits = model(ad, bd, prev)
            loss = NF.cross_entropy(logits.flatten(0, 1), target.flatten())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 1000 == 0:
                teacher_err = (logits.argmax(2) != target).any(1).sum().item()
                print(step, f'loss={loss.item():.6g}', f'train_exact_err={teacher_err}/{batch}', flush=True)
            if step >= 16000 and step % 2000 == 0:
                model.eval()
                scores = [ar_errors(model, 16384, device, k) for k in ('uniform','carry','noncarry','sparse')]
                score = scores[0] * 4 + sum(scores[1:])
                print('validation', step, scores, 'score', score, flush=True)
                if score <= best_score:
                    best_score = score; best = copy.deepcopy(model.state_dict())
                    torch.save(best, '/workspace/best.pt')
                model.train()
    if best is not None: model.load_state_dict(best)
    model.eval()
    print('FINAL checkpoint score', best_score, flush=True)
    for kind, n in [('uniform',262144),('carry',131072),('noncarry',131072),('sparse',131072),('repeat',131072)]:
        print(kind, ar_errors(model, n, device, kind), '/', n, flush=True)
    export(model, '/workspace/submission.py')


if __name__ == '__main__': main()

# Optional broad calibration after the core run.
def calibrate():
    torch.manual_seed(3110); random.seed(3110)
    device = torch.device('cuda')
    model = Model().to(device)
    model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.001, fused=True)
    batch = 4096
    best_state = copy.deepcopy(model.state_dict())
    best_score = sum(ar_errors(model, 32768, device, k) for k in ('uniform','carry','noncarry','sparse','repeat'))
    print('calibration initial', best_score, flush=True)
    for step in range(1, 6001):
        sizes = [batch // 2, batch // 8, batch // 8, batch // 8]
        sizes.append(batch - sum(sizes))
        vals = torch.randint(0, LIMIT, (sizes[0], 2), device=device)
        aa = [integer_digits(vals[:,0])]; bb = [integer_digits(vals[:,1])]
        for fn, n in ((lambda n,d: chains(n,d,True),sizes[1]),
                      (lambda n,d: chains(n,d,False),sizes[2]),
                      (sparse,sizes[3]), (repeated,sizes[4])):
            a,b=fn(n,device); aa.append(a);bb.append(b)
        ad,bd=torch.cat(aa),torch.cat(bb)
        order=torch.randperm(batch,device=device);ad,bd=ad[order],bd[order]
        target=labels(ad,bd)
        prev=torch.cat((torch.full((batch,1),10,device=device,dtype=torch.long),target[:,:-1]),1)
        opt.zero_grad(set_to_none=True)
        loss=NF.cross_entropy(model(ad,bd,prev).flatten(0,1),target.flatten())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%1000==0:
            model.eval(); scores=[ar_errors(model,32768,device,k) for k in ('uniform','carry','noncarry','sparse','repeat')]
            score=sum(scores);print('calibration',step,scores,score,flush=True)
            if score<=best_score: best_score=score;best_state=copy.deepcopy(model.state_dict())
            model.train()
    model.load_state_dict(best_state);model.eval();torch.save(best_state,'/workspace/best.pt')
    print('calibrated score',best_score,flush=True)
    for kind,n in [('uniform',524288),('carry',262144),('noncarry',262144),('sparse',262144),('repeat',262144)]:
        print('CAL',kind,ar_errors(model,n,device,kind),'/',n,flush=True)
    export(model,'/workspace/submission.py')

def final_boundary_calibration():
    torch.manual_seed(3111); random.seed(3111)
    device=torch.device('cuda');model=Model().to(device)
    model.load_state_dict(torch.load('/workspace/best.pt',weights_only=True));model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.001,fused=True)
    batch=4096
    for step in range(1,3001):
        n0=batch//2;n1=batch//8;n2=batch//8;n3=batch-n0-n1-n2
        vals=torch.randint(0,LIMIT,(n0,2),device=device)
        aa=[integer_digits(vals[:,0])];bb=[integer_digits(vals[:,1])]
        a,b=chains(n1,device,True);aa.append(a);bb.append(b)
        a,b=repeated(n2,device);aa.append(a);bb.append(b)
        a=torch.zeros((n3,15),dtype=torch.long,device=device);b=torch.zeros_like(a)
        rows=torch.arange(n3,device=device);p=torch.randint(0,14,(n3,),device=device)
        total=torch.randint(10,19,(n3,),device=device)
        da_low=torch.clamp(total-9,min=1); da=da_low+(torch.rand(n3,device=device)*(10-da_low)).long()
        # Explicitly reserve one quarter for 9+9, the discovered weak contrast.
        da=torch.where(rows%4==0,torch.full_like(da,9),da);total=torch.where(rows%4==0,torch.full_like(total,18),total)
        a[rows,p]=da;b[rows,p]=total-da;aa.append(a);bb.append(b)
        ad,bd=torch.cat(aa),torch.cat(bb);order=torch.randperm(batch,device=device);ad,bd=ad[order],bd[order]
        target=labels(ad,bd);prev=torch.cat((torch.full((batch,1),10,device=device,dtype=torch.long),target[:,:-1]),1)
        opt.zero_grad(set_to_none=True);loss=NF.cross_entropy(model(ad,bd,prev).flatten(0,1),target.flatten())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%1000==0:print('boundary calibration',step,loss.item(),flush=True)
    model.eval();torch.save(model.state_dict(),'/workspace/best.pt');export(model,'/workspace/submission.py')
