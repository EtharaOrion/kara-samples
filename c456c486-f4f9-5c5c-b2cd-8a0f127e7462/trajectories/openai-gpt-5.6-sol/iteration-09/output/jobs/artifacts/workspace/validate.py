import torch, random, time
from submission import AdditionTransformer
from train import batch_data
D='cuda'

def random_eval(width, count=1000000):
 m=AdditionTransformer(width=width).to(D); m.load_state_dict(torch.load(f'/workspace/model_w{width}.pt', weights_only=True)); m.eval()
 exact=bad=0; first=[]; start=time.time()
 with torch.no_grad():
  for _ in range(count//10000):
   x,y=batch_data(10000); p=m(x).argmax(-1); ok=p.eq(y).all(1); exact+=ok.sum().item()
   if len(first)<10:
    ids=(~ok).nonzero().flatten()[:10-len(first)]
    for i in ids: first.append((x[i].cpu(),y[i].cpu(),p[i].cpu()))
 print(width, exact/count, 'errors',count-exact,'seconds',time.time()-start)
 print('first output digit mismatches', [(y.ne(p)).nonzero().flatten().tolist() for _,y,p in first])
 return m

def encode(values):
 x=torch.zeros(len(values),15,2,dtype=torch.long,device=D); y=torch.zeros(len(values),15,dtype=torch.long,device=D)
 for row,(a,b) in enumerate(values):
  s=a+b
  for pos in range(14): a,x[row,pos,0]=divmod(a,10); b,x[row,pos,1]=divmod(b,10)
  for pos in range(15): s,y[row,pos]=divmod(s,10)
 return x,y

def structured(m):
 vals=[(0,0),(99999999999999,99999999999999),(99999999999999,1),(50000000000000,50000000000000),(12345678901234,87654321098766)]
 for n in range(1,14):
  ten=10**n
  vals += [(ten-1,1),(ten-1,ten-1),(10**14-ten,ten),(int(str(n%10)*14),int(str((9-n)%10)*14))]
 random.seed(44)
 for _ in range(100000):
  n=random.randrange(1,14); base=10**n
  vals.append((random.randrange(10**14//base)*base+base-1,random.randrange(1,10)))
 x,y=encode(vals)
 with torch.no_grad(): p=m(x).argmax(-1)
 ok=p.eq(y).all(1); print('structured',ok.float().mean().item(),'errors',(~ok).sum().item(),'of',len(vals))
 for i in (~ok).nonzero().flatten()[:20]: print('bad',vals[i],y[i].tolist(),p[i].tolist())

m=random_eval(10)
structured(m)
# Demonstrate QK scores change with digit content.
with torch.no_grad():
 x1=torch.zeros(1,15,2,dtype=torch.long,device=D); x2=x1.clone(); x2[:,5:,0]=7
 h1=m.digit(x1[...,0])+m.digit(x1[...,1])+m.position
 h2=m.digit(x2[...,0])+m.digit(x2[...,1])+m.position
 q1,k1,_=m.block.qkv(m.block.norm1(h1)).chunk(3,-1)
 q2,k2,_=m.block.qkv(m.block.norm1(h2)).chunk(3,-1)
 print('qk content delta',((q1@k1.transpose(-1,-2))-(q2@k2.transpose(-1,-2))).abs().max().item())
