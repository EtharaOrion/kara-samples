import ast
import copy
import random
import sys
from pathlib import Path
import torch
sys.path.insert(0, '/workspace')
import submission
import train

@torch.inference_mode()
def batch_pairs(model, pairs, device, chunk=8192):
    errors=[]
    for start in range(0,len(pairs),chunk):
        part=pairs[start:start+chunk]
        av=torch.tensor([a for a,b in part],device=device)
        bv=torch.tensor([b for a,b in part],device=device)
        ad=train.integer_digits(av);bd=train.integer_digits(bv)
        prev=torch.full((len(part),1),10,dtype=torch.long,device=device)
        out=[]
        for _ in range(15):
            d=model(ad,bd,prev)[:,-1].argmax(1);out.append(d);prev=torch.cat((prev,d[:,None]),1)
        got=(torch.stack(out,1)*train.POW10.to(device)).sum(1).cpu().tolist()
        for pair,value in zip(part,got):
            if value != pair[0]+pair[1]: errors.append((pair,value))
    return errors

device=torch.device('cuda')
m=train.Model().to(device);m.load_state_dict(torch.load('/workspace/best.pt',weights_only=True));m.eval()
print('million uniform errors',train.ar_errors(m,1048576,device,'uniform'))
print('half-million carry errors',train.ar_errors(m,524288,device,'carry'))
print('half-million pattern errors',train.ar_errors(m,524288,device,'repeat'))
pairs={(0,0),(99_999_999_999_999,0),(99_999_999_999_999,1),(99_999_999_999_999,99_999_999_999_999)}
for p in range(14):
    q=10**p
    candidates=[0,1,4,5,8,9,q-1,q,q+1,5*q,9*q,10*q-1]
    for x in candidates:
        if 0<=x<100_000_000_000_000:
            for y in (0,1,5,9,q-1,q,min(99_999_999_999_999,10*q-x)):
                if 0<=y<100_000_000_000_000:pairs.add((x,y));pairs.add((y,x))
for length in range(1,15):
    run=10**length-1
    for start in range(15-length):
        x=run*10**start
        if x<100_000_000_000_000:
            for y in (1*10**start,5*10**start,9*10**start):
                pairs.add((x,y));pairs.add((y,x))
pairs=sorted(pairs)
errs=batch_pairs(m,pairs,device)
print('systematic errors',len(errs),'/',len(pairs),errs[:10])
# First-block normalized content produces genuinely different QK scores.
ad1=train.integer_digits(torch.tensor([12345678901234],device=device));bd1=train.integer_digits(torch.tensor([9876543210123],device=device))
ad2=train.integer_digits(torch.tensor([11111111111111],device=device));bd2=train.integer_digits(torch.tensor([22222222222222],device=device))
prev=torch.full((1,1),10,dtype=torch.long,device=device)
def scores(ad,bd):
    src=m.a_embed(ad)+m.b_embed(bd)+m.out_embed.weight[10]
    absent=torch.full((1,1),10,dtype=torch.long,device=device)
    x=torch.cat((src,m.a_embed(absent)+m.b_embed(absent)+m.out_embed(prev)),1)+m.position[:16]
    q,k,_=m.blocks[0].qkv(m.blocks[0].n1(x)).chunk(3,-1)
    return torch.matmul(q.view(1,16,3,3).transpose(1,2),k.view(1,16,3,3).transpose(1,2).transpose(-2,-1))
print('qk max input difference',float((scores(ad1,bd1)-scores(ad2,bd2)).abs().max()))
# Actual exported CPU interface and model sensitivity.
sm,meta=submission.build_model();print('submission params',sum(p.numel() for p in sm.parameters()),meta)
examples=[(0,0),(12,34),(99999999999999,1),(50000000000000,49999999999999),(99999999999999,99999999999999)]
print('cpu examples',[(a,b,submission.add(sm,a,b),a+b) for a,b in examples])
base=submission.add(sm,12345678901234,87654321098765)
with torch.no_grad(): sm.head.weight.zero_();sm.head.bias.zero_()
changed=submission.add(sm,12345678901234,87654321098765)
print('perturbation',base,changed,base!=changed)
tree=ast.parse(Path('/workspace/submission.py').read_text())
print('imports',[ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))])
