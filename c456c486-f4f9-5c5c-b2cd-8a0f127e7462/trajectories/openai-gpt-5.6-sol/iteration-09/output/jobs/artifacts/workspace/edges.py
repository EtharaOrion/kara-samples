import torch, random
from submission import AdditionTransformer
D='cuda'
random.seed(44)
vals=[(0,0),(99999999999999,99999999999999),(99999999999999,1),(50000000000000,50000000000000),(12345678901234,87654321098766)]
for n in range(1,14):
 t=10**n
 vals += [(t-1,1),(t-1,t-1),(10**14-t,t),(int(str(n%10)*14),int(str((9-n)%10)*14))]
for _ in range(20000):
 n=random.randrange(1,14); base=10**n
 vals.append((random.randrange(10**14//base)*base+base-1,random.randrange(1,10)))
a=torch.tensor([v[0] for v in vals],device=D); b=torch.tensor([v[1] for v in vals],device=D); s=a+b
x=torch.zeros(len(vals),15,2,dtype=torch.long,device=D); y=torch.zeros(len(vals),15,dtype=torch.long,device=D)
for pos in range(15):
 if pos<14:
  x[:,pos,0]=a%10; x[:,pos,1]=b%10; a//=10; b//=10
 y[:,pos]=s%10; s//=10
for width in (10,12):
 m=AdditionTransformer(width=width).to(D); m.load_state_dict(torch.load(f'/workspace/model_w{width}.pt',weights_only=True)); m.eval()
 with torch.no_grad(): p=m(x).argmax(-1)
 ok=p.eq(y).all(1); print(width,ok.float().mean().item(),int((~ok).sum()),'of',len(vals))
 for i in (~ok).nonzero().flatten()[:10]: print(vals[i],y[i].tolist(),p[i].tolist())
 with torch.no_grad():
  z=torch.zeros(1,15,2,dtype=torch.long,device=D); z2=z.clone();z2[:,5:,0]=7
  h1=m.digit(z[...,0])+m.digit(z[...,1])+m.position;h2=m.digit(z2[...,0])+m.digit(z2[...,1])+m.position
  q1,k1,_=m.block.qkv(m.block.norm1(h1)).chunk(3,-1);q2,k2,_=m.block.qkv(m.block.norm1(h2)).chunk(3,-1)
  print('qk-delta',((q1@k1.transpose(-1,-2))-(q2@k2.transpose(-1,-2))).abs().max().item())
