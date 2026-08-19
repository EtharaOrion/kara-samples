import copy, sys, time, torch
sys.path.insert(0,'/workspace')
from submission import Model
from train_ar import uni, structured, digs, ev, export, P, LIMIT
D='cuda'; B=4096
m=Model().cuda().train(); m.load_state_dict(torch.load('/workspace/refined.pt',weights_only=True))
opt=torch.optim.AdamW(m.parameters(),lr=5e-6,betas=(.9,.98),weight_decay=.0005)
def sparse(n):
 a=torch.zeros(n,dtype=torch.long,device=D); b=torch.zeros_like(a)
 st=torch.randint(0,14,(n,),device=D); ln=1+(torch.rand(n,device=D)*(14-st)).long(); block=P[ln]-1; p=P[st]
 # short all-9 blocks, random blocks, and isolated ordinary decimal chunks
 f=torch.randint(0,4,(n,),device=D)
 vals=torch.randint(0,1000,(n,),device=D)
 vals=torch.minimum(vals,P[ln]-1)
 vals=torch.where(f==0,block,vals); vals=torch.where(f==1,torch.minimum(torch.full_like(vals,99),block),vals)
 a=vals*p
 idx=f==3; b[idx]=P[st[idx]]
 swap=torch.rand(n,device=D)<.5; aa=torch.where(swap,b,a); bb=torch.where(swap,a,b)
 return aa.clamp_max(LIMIT-1),bb.clamp_max(LIMIT-1)
def edges(model):
 model.eval(); fail=0
 cases=[]
 for k in range(14):
  p=10**k
  for v in [9,10,11,19,90,98,99,100,101,109,999]:
   if v*p<LIMIT:
    cases += [(v*p,0),(0,v*p),(v*p,p)]
 for a,b in cases:
  ad=digs(torch.tensor([a],device=D)); bd=digs(torch.tensor([b],device=D)); prev=torch.empty(1,0,dtype=torch.long,device=D)
  for _ in range(15): prev=torch.cat((prev,model(ad,bd,prev)[:,-1].argmax(-1,keepdim=True)),1)
  val=int(sum(int(prev[0,i])*10**i for i in range(15)))
  fail += val != a+b
 model.train(); return fail,len(cases)
best=copy.deepcopy(m.state_dict()); bestscore=10**9; t=time.time()
for s in range(30000):
 a0,b0=uni(B//2); a1,b1=structured(B//4); a2,b2=sparse(B-B//2-B//4); a=torch.cat((a0,a1,a2)); b=torch.cat((b0,b1,b2)); y=digs(a+b)
 z=m(digs(a),digs(b),y[:,:-1])[:,14:]; loss=torch.nn.functional.cross_entropy(z.reshape(-1,10),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
 if (s+1)%5000==0:
  eu=ev(m,131072); es=ev(m,131072,'structured'); ee=edges(m); score=eu[0]*4+es[0]+ee[0]*10000; print(s+1,float(loss),eu,es,ee,score,'sec',round(time.time()-t),flush=True)
  if score<bestscore: bestscore=score; best=copy.deepcopy(m.state_dict()); torch.save(best,'/workspace/calibrated.pt')
m.load_state_dict(best); print('final-u',ev(m,1048576),flush=True); print('final-s',ev(m,524288,'structured'),flush=True); print('edges',edges(m),flush=True); export(m)
