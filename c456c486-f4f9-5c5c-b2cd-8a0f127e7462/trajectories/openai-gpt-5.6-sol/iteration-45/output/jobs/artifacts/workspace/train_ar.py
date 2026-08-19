import copy, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace")
from submission import Model

D="cuda"; B=4096; LIMIT=100_000_000_000_000
P=torch.tensor([10**i for i in range(16)],device=D,dtype=torch.long)
def digs(x): return (x[:,None]//P[None,:15])%10
def uni(n): return torch.randint(0,LIMIT,(n,),device=D),torch.randint(0,LIMIT,(n,),device=D)
def structured(n):
 a,b=uni(n); f=torch.randint(0,8,(n,),device=D); st=torch.randint(0,14,(n,),device=D); ln=1+(torch.rand(n,device=D)*(14-st)).long(); p=P[st]; run=P[ln]-1
 idx=f==0; a[idx]=run[idx]*p[idx]; b[idx]=p[idx]
 idx=f==1; a[idx]=(run[idx]-1).clamp_min(0)*p[idx]; b[idx]=p[idx]
 idx=f==2; da=torch.randint(0,10,(n,),device=D); db=torch.randint(0,10,(n,),device=D); special=torch.rand(n,device=D)<.6; da[special]=torch.where(torch.rand(n,device=D)[special]<.5,5,9); db[special]=da[special]; a[idx]=da[idx]*p[idx]; b[idx]=db[idx]*p[idx]
 idx=f==3; rep=(LIMIT-1)//9; a[idx]=torch.randint(0,10,(n,),device=D)[idx]*rep; b[idx]=torch.randint(0,10,(n,),device=D)[idx]*rep
 idx=f==4; a[idx]=LIMIT-1; b[idx]=p[idx]
 idx=f==5; low=1+(torch.rand(n,device=D)*run).long(); a[idx]=low[idx]*p[idx]; b[idx]=(P[ln[idx]]-low[idx])*p[idx]
 idx=f==6; # shifted equal-9 or 5 boundaries with random lower noise
 a[idx]=torch.where(torch.rand(n,device=D)[idx]<.5,5*p[idx],9*p[idx]); b[idx]=a[idx]
 return a,b
def make(step):
 if step<36000: a,b=uni(B)
 else:
  a0,b0=uni(B//2); a1,b1=structured(B-B//2); a=torch.cat((a0,a1)); b=torch.cat((b0,b1))
 y=digs(a+b); return digs(a),digs(b),y
@torch.no_grad()
def ev(m,count,kind="uniform",chunk=8192):
 m.eval(); err=de=0
 while count:
  n=min(count,chunk); count-=n; a,b=uni(n) if kind=="uniform" else structured(n); y=digs(a+b)
  # batched autoregressive validation
  prev=torch.empty(n,0,dtype=torch.long,device=D)
  for _ in range(15): prev=torch.cat((prev,m(digs(a),digs(b),prev)[:,-1].argmax(-1,keepdim=True)),1)
  bad=prev!=y; err+=bad.any(1).sum().item(); de+=bad.sum().item()
 m.train(); return err,de
def export(m):
 src=Path('/workspace/submission.py').read_text(); prefix=src[:src.index('def build_model():')]
 state={k:v.detach().float().cpu().tolist() for k,v in m.state_dict().items()}
 tail='''def build_model():\n    model = Model()\n    model.load_state_dict({k: torch.tensor(v) for k, v in STATE.items()})\n    model.eval()\n    return model, {"architecture": "aligned autoregressive causal transformer", "digits": 14}\n\n\ndef add(model, a: int, b: int) -> int:\n    device = next(model.parameters()).device\n    ad = torch.tensor([[int(c) for c in f"{a:014d}"[::-1]] + [0]], device=device)\n    bd = torch.tensor([[int(c) for c in f"{b:014d}"[::-1]] + [0]], device=device)\n    previous = torch.empty((1, 0), dtype=torch.long, device=device)\n    with torch.no_grad():\n        for _ in range(15):\n            digit = model(ad, bd, previous)[:, -1].argmax(-1, keepdim=True)\n            previous = torch.cat((previous, digit), dim=1)\n    text = "".join(str(d) for d in reversed(previous[0].tolist())).lstrip("0")\n    return int(text or "0")\n'''
 Path('/workspace/submission.py').write_text(prefix+'STATE = '+repr(state)+'\n\n'+tail)
def main():
 torch.manual_seed(45002); random.seed(45002); torch.set_float32_matmul_precision('high'); m=Model().to(D).train(); print('params',sum(p.numel() for p in m.parameters()),flush=True)
 opt=torch.optim.AdamW(m.parameters(),lr=3e-3,betas=(.9,.98),weight_decay=.003); total=int(sys.argv[1]) if len(sys.argv)>1 else 120000; best=None; bs=10**9; t=time.time()
 for s in range(total):
  if s==12000: opt.param_groups[0]['lr']=1e-3
  if s==36000: opt.param_groups[0]['lr']=3e-4
  if s==68000: opt.param_groups[0]['lr']=1e-4
  if s==96000: opt.param_groups[0]['lr']=3e-5
  if s==112000: opt.param_groups[0]['lr']=1e-5
  a,b,y=make(s); logits=m(a,b,y[:,:-1])[:,14:]; loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
  if (s+1)%4000==0: print(s+1,float(loss),opt.param_groups[0]['lr'],'sec',round(time.time()-t),flush=True)
  if s+1>=36000 and (s+1)%12000==0:
   eu=ev(m,65536); es=ev(m,65536,'structured'); score=eu[0]*4+es[0]; print('eval',s+1,eu,es,score,flush=True)
   if score<=bs: bs=score; best=copy.deepcopy(m.state_dict()); torch.save(best,'/workspace/best_ar.pt')
 if best: m.load_state_dict(best)
 print('final-u',ev(m,1048576),flush=True); print('final-s',ev(m,524288,'structured'),flush=True); export(m)
if __name__=='__main__': main()
