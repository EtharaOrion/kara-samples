import torch, random, sys
from train import Adder
m=Adder(False);m.load_state_dict(torch.load('/workspace/unshared.pt',weights_only=True));m.eval();torch.set_num_threads(16)
W=5;N=int(sys.argv[1]) if len(sys.argv)>1 else 1000;random.seed(271828);good=0

def ds(x):
 r=[]
 for _ in range(15):r.append(x%10);x//=10
 return r

def candidates(aa,bb):
 beams=[([],torch.tensor(0.))]
 with torch.no_grad():
  for c in range(15):
   seqs=[q[0]+[aa[c],bb[c]] for q in beams]; lp=m(torch.tensor(seqs))[:,-1].log_softmax(-1); nxt=[]
   for j,(old,s) in enumerate(beams):
    v,d=lp[j].topk(W)
    for k in range(W): nxt.append((seqs[j]+[int(d[k])],torch.stack((s,v[k])).sum()))
   nxt.sort(key=lambda q:float(q[1]),reverse=True);beams=nxt[:W]
 return [q[0][2::3] for q in beams]

def score(aa,bb,outs):
 rows=[]
 for yy in outs: rows.append([z for c in range(15) for z in (aa[c],bb[c],yy[c])])
 t=torch.tensor(rows); lp=m(t[:,:-1]).log_softmax(-1); pos=torch.arange(1,44,3)
 return lp[:,pos].gather(2,t[:,2::3,None]).squeeze(-1).sum(1)
with torch.no_grad():
 for i in range(N):
  a=random.randrange(10**14);b=random.randrange(10**14);aa=ds(a);bb=ds(b)
  outs=candidates(aa,bb); outs.extend(candidates(bb,aa))
  s=torch.stack((score(aa,bb,outs),score(bb,aa,outs))).sum(0); yy=outs[int(s.argmax())]
  val=0
  for d in reversed(yy):val=val*10+d
  good+=val==a+b
  if (i+1)%100==0:print(i+1,good/(i+1),flush=True)
