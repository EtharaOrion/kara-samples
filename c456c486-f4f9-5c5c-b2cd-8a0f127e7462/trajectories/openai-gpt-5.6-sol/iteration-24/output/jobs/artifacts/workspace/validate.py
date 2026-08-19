import sys, time, torch
sys.path.insert(0,"/workspace")
from train import Model, digits, carry_examples, patterned, BASE, POW

torch.manual_seed(240024)
model=Model().cuda().eval(); model.load_state_dict(torch.load('/workspace/final.pt',weights_only=True))

@torch.no_grad()
def ar(a,b):
    ad=digits(a,14); bd=digits(b,14); prev=torch.empty((a.shape[0],0),dtype=torch.long,device='cuda')
    for _ in range(15): prev=torch.cat((prev,model(ad,bd,prev)[:,-1].argmax(-1,keepdim=True)),1)
    return (prev*POW.to('cuda')).sum(1)

@torch.no_grad()
def run(n,mode,bs=16384):
    err=0; first=[]; t=time.time()
    for off in range(0,n,bs):
        k=min(bs,n-off)
        if mode=='random': a=torch.randint(0,BASE,(k,),device='cuda'); b=torch.randint(0,BASE,(k,),device='cuda')
        elif mode=='carry': a,b=carry_examples(k,'cuda')
        else:a,b=patterned(k,'cuda')
        got=ar(a,b); bad=got.ne(a+b); err+=bad.sum().item()
        if bad.any() and len(first)<10:
            ix=bad.nonzero().flatten()[:10-len(first)]
            first += list(zip(a[ix].tolist(),b[ix].tolist(),got[ix].tolist(),(a[ix]+b[ix]).tolist()))
    print(mode,n,'errors',err,'seconds',time.time()-t,'first',first,flush=True)

# Systematic long carries/non-carries, boundaries, repeats, sparse positions and swaps.
pairs={(0,0),(BASE-1,0),(BASE-1,1),(BASE-1,BASE-1),(1,BASE-1)}
for p in [10**i for i in range(15)]:
    if p<=BASE:
        vals={0,1,max(0,p-2),max(0,p-1),min(BASE-1,p),min(BASE-1,p+1)}
        for x in vals:
            for y in [0,1,2,8,9,10,11,max(0,p-x),max(0,p-1-x)]:
                if 0<=x<BASE and 0<=y<BASE: pairs.add((x,y));pairs.add((y,x))
for d in range(10):
 for e in range(10): pairs.add((d*11111111111111,e*11111111111111))
for start in range(14):
 for length in range(1,15-start):
    unit=10**start; carry_run=(10**length-1)*unit
    for x in [1,2,5,9]:
      if carry_run<BASE: pairs.add((carry_run,x*unit));pairs.add((x*unit,carry_run))
a=torch.tensor([x for x,y in pairs],device='cuda');b=torch.tensor([y for x,y in pairs],device='cuda')
got=ar(a,b); bad=got.ne(a+b)
print('systematic',len(pairs),'errors',bad.sum().item(),[(a[i].item(),b[i].item(),got[i].item(),(a[i]+b[i]).item()) for i in bad.nonzero().flatten()[:20]],flush=True)
run(1048576,'random'); run(524288,'carry'); run(524288,'pattern')
# Attention content dependence before masking.
with torch.no_grad():
 ad1=digits(torch.tensor([12345678901234],device='cuda'),14);bd1=digits(torch.tensor([55555555555555],device='cuda'),14)
 ad2=digits(torch.tensor([98765432109876],device='cuda'),14);bd2=digits(torch.tensor([11111111111111],device='cuda'),14)
 x1=model.a_emb(ad1)+model.b_emb(bd1)+model.pos[:14];x2=model.a_emb(ad2)+model.b_emb(bd2)+model.pos[:14]
 z1=model.blocks[0].n1(x1);z2=model.blocks[0].n1(x2)
 q1,k1,_=model.blocks[0].qkv(z1).chunk(3,-1);q2,k2,_=model.blocks[0].qkv(z2).chunk(3,-1)
 s1=q1@k1.transpose(-1,-2);s2=q2@k2.transpose(-1,-2)
 print('qk_max_change',(s1-s2).abs().max().item())
