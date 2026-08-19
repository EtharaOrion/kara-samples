import sys
import torch
sys.path.insert(0, '/workspace')
import submission
from train import uniform_batch, structured_batch, make_tokens, autoregressive_errors, DIGITS

device=torch.device('cuda')
powers=10 ** torch.arange(DIGITS,device=device,dtype=torch.long)
model,meta=submission.build_model(); model=model.to(device).eval()
print('parameters',sum(p.numel() for p in model.parameters()),meta)
for kind in ('uniform','structured'):
    e,n=autoregressive_errors(model,128,device,powers,kind)
    print(kind,e,n,'accuracy',1-e/n)
# Systematic powers, maximal carries/non-carries, repeated digits, and boundaries.
cases={(0,0),(99_999_999_999_999,0),(99_999_999_999_999,1),(99_999_999_999_999,99_999_999_999_999)}
for k in range(14):
    p=10**k
    for x,y in ((p,1),(9*p,9*p),(5*p,5*p),((p-1) if k else 0,1),((10*p-1),1),((10*p-1),p),(99_999_999_999_999-p+1,p-1)):
        if 0<=x<100_000_000_000_000 and 0<=y<100_000_000_000_000: cases.add((x,y)); cases.add((y,x))
for d in range(10):
    x=int(str(d)*14)
    for e in range(10): cases.add((x,int(str(e)*14)))
a=torch.tensor([x for x,y in cases],device=device); b=torch.tensor([y for x,y in cases],device=device)
ad,bd,target=make_tokens(a,b)
generated=torch.empty((len(cases),0),dtype=torch.long,device=device)
with torch.no_grad():
    for _ in range(15): generated=torch.cat((generated,model(ad,bd,generated)[:,-1].argmax(-1,keepdim=True)),1)
bad=(generated!=target).any(1)
print('systematic',bad.sum().item(),len(cases))
if bad.any():
    indices=bad.nonzero()[:20,0].tolist(); pairs=list(cases)
    for i in indices: print('FAIL',pairs[i],generated[i].tolist(),target[i].tolist())
# QK scores must differ with content.
with torch.no_grad():
    z1=model.block1.norm(model.a_embedding(ad[:1])+model.b_embedding(bd[:1])+model.position[:15])
    z2=model.block1.norm(model.a_embedding(ad[-1:])+model.b_embedding(bd[-1:])+model.position[:15])
    q1,k1,_=model.block1.qkv(z1).chunk(3,-1); q2,k2,_=model.block1.qkv(z2).chunk(3,-1)
    s1=q1.view(1,15,3,3).transpose(1,2)@k1.view(1,15,3,3).transpose(1,2).transpose(-2,-1)
    s2=q2.view(1,15,3,3).transpose(1,2)@k2.view(1,15,3,3).transpose(1,2).transpose(-2,-1)
print('qk_delta',(s1-s2).abs().max().item())
