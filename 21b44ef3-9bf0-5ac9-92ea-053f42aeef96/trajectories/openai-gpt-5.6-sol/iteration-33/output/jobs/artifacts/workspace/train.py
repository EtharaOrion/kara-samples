import argparse
import copy
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

D, H, DH, NPOS = 20, 4, 5, 25
LOW, HIGH = 10_000_000, 99_999_999
ROOT = Path('/workspace')


class TrainModel(nn.Module):
    def __init__(self, ff=4):
        super().__init__()
        self.ff = ff
        self.token = nn.Embedding(11, D)
        self.pos = nn.Parameter(torch.empty(NPOS, 2))
        self.pos_basis = nn.Parameter(torch.empty(2, D))
        self.norm = nn.LayerNorm(D)
        self.q = nn.ModuleList([nn.Linear(D, D, False) for _ in range(2)])
        self.k = nn.Linear(D, DH, False)
        self.v = nn.Linear(D, DH, False)
        self.o = nn.ModuleList([nn.Linear(D, D, False) for _ in range(2)])
        self.ff1 = nn.Linear(D, ff, True)
        self.ff2 = nn.Linear(ff, D, False)
        self.head = nn.Linear(D, 10, True)
        self.register_buffer('mask', torch.triu(torch.full((NPOS, NPOS), float('-inf')), 1), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.normal_(p, std=.12) if p.ndim > 1 else nn.init.zeros_(p)
        nn.init.ones_(self.norm.weight)

    def forward(self, tok):
        t = tok.shape[1]
        x = self.token(tok) + self.pos[:t] @ self.pos_basis
        for i in range(2):
            n = self.norm(x)
            q = self.q[i](n).view(-1, t, H, DH)
            k, v = self.k(n), self.v(n)
            s = torch.einsum('bthd,bsd->bhts', q, k) / math.sqrt(DH)
            a = F.softmax(s + self.mask[:t, :t], -1)
            x = x + self.o[i](torch.einsum('bhts,bsd->bthd', a, v).reshape(-1, t, D))
            x = x + self.ff2(F.gelu(self.ff1(F.layer_norm(x, (D,)))))
        return self.head(F.layer_norm(x, (D,)))


def digits(x):
    out = []
    for _ in range(9):
        out.append(x.remainder(10))
        x = x.div(10, rounding_mode='floor')
    return torch.stack(out, 1)


def uniform(n, dev):
    return torch.randint(LOW, HIGH + 1, (n,), device=dev), torch.randint(LOW, HIGH + 1, (n,), device=dev)


def structured(n, dev):
    # A broad mixture, with randomized leading digits and every possible carry-run length.
    a, b = uniform(n, dev)
    kind = torch.randint(0, 9, (n,), device=dev)
    pow10 = torch.tensor([1,10,100,1000,10000,100000,1000000,10000000], device=dev)
    run = torch.randint(1, 8, (n,), device=dev)
    p = pow10[run]

    # Complements around 10^8, perturbed on both sides.
    m = kind == 0
    aa = torch.randint(LOW, HIGH + 1, (n,), device=dev)
    delta = torch.randint(-20, 21, (n,), device=dev)
    bb = 100_000_000 - aa + delta
    a[m], b[m] = aa[m], bb.clamp(LOW, HIGH)[m]

    # Unequal suffix carry chains: prefix is fully random rather than a fixed edge.
    m = kind == 1
    basea = torch.randint(LOW, HIGH + 1, (n,), device=dev)
    baseb = torch.randint(LOW, HIGH + 1, (n,), device=dev)
    suffix = p - torch.randint(1, 10, (n,), device=dev)
    aa = basea.div(p, rounding_mode='floor') * p + suffix
    bb = baseb.div(p, rounding_mode='floor') * p + torch.randint(1, 10, (n,), device=dev)
    a[m], b[m] = aa.clamp(LOW,HIGH)[m], bb.clamp(LOW,HIGH)[m]

    # Decimal boundaries on arbitrary scales and both sides.
    m = kind == 2
    center = torch.randint(1, 100_000_000, (n,), device=dev).div(p, rounding_mode='floor') * p
    aa = center + torch.randint(-20, 21, (n,), device=dev)
    bb = torch.randint(LOW, HIGH + 1, (n,), device=dev)
    a[m], b[m] = aa.clamp(LOW,HIGH)[m], bb[m]

    # Repeated digits.
    m = kind == 3
    da = torch.randint(1, 10, (n,), device=dev)
    db = torch.randint(1, 10, (n,), device=dev)
    aa, bb = da * 11_111_111, db * 11_111_111
    a[m], b[m] = aa[m], bb[m]

    # Sparse interior decimal digits, always preserving an 8-digit leading digit.
    m = kind == 4
    lead = torch.randint(1,10,(n,),device=dev) * 10_000_000
    place = pow10[torch.randint(0,7,(n,),device=dev)]
    aa = lead + torch.randint(0,10,(n,),device=dev)*place + torch.randint(0,10,(n,),device=dev)
    a[m] = aa[m]

    # Near extrema.
    m = kind == 5
    aa = torch.where(torch.rand(n,device=dev)<.5, LOW + torch.randint(0,10000,(n,),device=dev), HIGH-torch.randint(0,10000,(n,),device=dev))
    bb = torch.where(torch.rand(n,device=dev)<.5, LOW + torch.randint(0,10000,(n,),device=dev), HIGH-torch.randint(0,10000,(n,),device=dev))
    a[m], b[m] = aa[m], bb[m]

    # Long zero/nine suffixes with independently random other operand.
    m = kind == 6
    pref = torch.randint(1, 100_000_000, (n,), device=dev).div(p, rounding_mode='floor')
    aa = pref*p + torch.where(torch.rand(n,device=dev)<.5, torch.zeros(n,device=dev,dtype=torch.long), p-1)
    a[m] = aa.clamp(LOW,HIGH)[m]

    # Carry begins/stops at independently selected columns.
    m = kind == 7
    r2 = torch.randint(1,8,(n,),device=dev); p2=pow10[r2]
    aa = torch.randint(LOW,HIGH+1,(n,),device=dev).div(p2,rounding_mode='floor')*p2 + p2-1
    bb = torch.randint(LOW,HIGH+1,(n,),device=dev).div(p,rounding_mode='floor')*p + 1
    a[m], b[m] = aa.clamp(LOW,HIGH)[m], bb.clamp(LOW,HIGH)[m]
    return a, b


def batch(n, dev, structured_fraction=.45):
    a, b = uniform(n, dev)
    count = int(n * structured_fraction)
    sa, sb = structured(count, dev)
    a[:count], b[:count] = sa, sb
    ad, bd, yd = digits(a)[:,:8], digits(b)[:,:8], digits(a+b)
    tok = torch.empty(n, 25, dtype=torch.long, device=dev)
    tok[:,0:16:2], tok[:,1:16:2], tok[:,16] = ad, bd, 10
    tok[:,17:] = yd[:,:8]
    return tok, yd


def evaluate(model, total=100000, structured_only=False, batch_size=8192):
    model.eval(); good=seen=0; minmargin=100.
    with torch.no_grad():
        while seen < total:
            n=min(batch_size,total-seen)
            a,b = structured(n,'cuda') if structured_only else uniform(n,'cuda')
            ad,bd,y=digits(a)[:,:8],digits(b)[:,:8],digits(a+b)
            tok=torch.empty(n,17,dtype=torch.long,device='cuda'); tok[:,0:16:2]=ad;tok[:,1:16:2]=bd;tok[:,16]=10
            pred=[]
            for j in range(9):
                logits=model(tok)[:,-1]; d=logits.argmax(1); pred.append(d)
                true=logits.gather(1,y[:,j:j+1]).squeeze(1)
                alt=logits.masked_fill(F.one_hot(y[:,j],10).bool(),-1e9).max(1).values
                minmargin=min(minmargin,float((true-alt).min()))
                if j<8: tok=torch.cat((tok,d[:,None]),1)
            good += int((torch.stack(pred,1)==y).all(1).sum()); seen+=n
    model.train(); return good,seen,minmargin


def train_stage(model, steps, lr, sf, label, batch_size=8192):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
    for step in range(1,steps+1):
        tok,y=batch(batch_size,'cuda',sf)
        logits=model(tok)[:,16:25]
        loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%2000==0 or step==steps:
            print(label,step,'loss',float(loss),flush=True)
            torch.save(model.state_dict(),ROOT/f'{label}.pt')
    return model


def prune(model, width):
    score=model.ff1.weight.norm(dim=1)*model.ff2.weight.norm(dim=0)
    keep=score.topk(width).indices.sort().values
    out=TrainModel(width).cuda()
    state=model.state_dict(); target=out.state_dict()
    for k in target:
        if k=='ff1.weight': target[k].copy_(state[k][keep])
        elif k=='ff1.bias': target[k].copy_(state[k][keep])
        elif k=='ff2.weight': target[k].copy_(state[k][:,keep])
        else: target[k].copy_(state[k])
    return out


def gauge_export(model):
    model=copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        # Affine positional gauge with anchors 0,17,22.
        C=model.pos; B=model.pos_basis
        origin=C[0].clone(); R=torch.stack((C[17]-origin,C[22]-origin),0)
        model.token.weight.add_(origin @ B)
        Cnew=(C-origin) @ torch.linalg.inv(R)
        Bnew=R @ B
        free=torch.cat((Cnew[1:17],Cnew[18:22],Cnew[23:]),0)
        # Gauge-fix K and V first five input columns to identity.
        Sk=model.k.weight[:,:5].clone()
        Knew=torch.linalg.solve(Sk,model.k.weight)
        for q in model.q:
            blocks=q.weight.reshape(H,DH,D)
            q.weight.copy_(torch.einsum('ab,hbc->hac',Sk.T,blocks).reshape(D,D))
        Sv=model.v.weight[:,:5].clone()
        Vnew=torch.linalg.solve(Sv,model.v.weight)
        for o in model.o:
            blocks=o.weight.reshape(D,H,DH)
            o.weight.copy_(torch.einsum('ohd,de->ohe',blocks,Sv).reshape(D,D))
        scalar=model.token.weight[0,0].clone()
        model.token.weight.sub_(scalar)
        token_free=model.token.weight.flatten()[1:]
        params=[token_free,free,Bnew,Knew[:,5:],Vnew[:,5:],model.norm.weight,
                model.norm.bias,model.q[0].weight,model.q[1].weight,model.o[0].weight,
                model.o[1].weight,model.ff1.weight,model.ff1.bias,model.ff2.weight,
                model.head.weight,model.head.bias]
    return params


def write_submission(model):
    params=gauge_export(model)
    template=(ROOT/'submission.py').read_text()
    start=template.index('_WEIGHTS =')
    end=template.index('\n\n\ndef build_model',start)
    text='_WEIGHTS = [\n'+''.join('    '+repr(p.flatten().tolist())+',\n' for p in params)+']'
    template=template[:start]+text+template[end:]
    (ROOT/'submission.py').write_text(template)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--resume'); args=ap.parse_args()
    torch.manual_seed(33031); torch.backends.cuda.matmul.allow_tf32=True
    if args.resume:
        data=torch.load(args.resume,map_location='cuda',weights_only=True)
        width=data['ff1.weight'].shape[0]; model=TrainModel(width).cuda(); model.load_state_dict(data)
    else:
        model=TrainModel(4).cuda()
        train_stage(model,36000,2e-3,.42,'teacher')
        train_stage(model,6000,8e-5,.50,'stable')
        print('teacher eval',evaluate(model,200000),evaluate(model,200000,True),flush=True)
        model=prune(model,3); train_stage(model,18000,2e-5,.52,'width3')
        print('w3 eval',evaluate(model,200000),evaluate(model,200000,True),flush=True)
        model=prune(model,2)
    train_stage(model,30000,1e-5,.52,'width2a')
    train_stage(model,16000,3e-6,.60,'width2b')
    print('final eval',evaluate(model,500000),evaluate(model,500000,True),flush=True)
    torch.save(model.state_dict(),ROOT/'final.pt'); write_submission(model)

if __name__=='__main__': main()
