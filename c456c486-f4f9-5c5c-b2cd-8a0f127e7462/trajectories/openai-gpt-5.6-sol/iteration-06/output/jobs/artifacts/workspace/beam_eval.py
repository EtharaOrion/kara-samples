import torch, random, sys
from train import Adder
m=Adder(False); m.load_state_dict(torch.load('/workspace/unshared.pt',weights_only=True)); m.eval(); torch.set_num_threads(16)
W=int(sys.argv[1]) if len(sys.argv)>1 else 5; N=int(sys.argv[2]) if len(sys.argv)>2 else 2000
random.seed(71828); good=0
with torch.no_grad():
 for z in range(N):
  a=random.randrange(10**14); b=random.randrange(10**14); aa=[];bb=[]; x=a;y=b
  for _ in range(15): aa.append(x%10);x//=10;bb.append(y%10);y//=10
  beams=[([],0.0)]
  for c in range(15):
   seqs=[q[0]+[aa[c],bb[c]] for q in beams]
   logits=m(torch.tensor(seqs))[:,-1].log_softmax(-1)
   candidates=[]
   for j,(_,score) in enumerate(beams):
    vals,inds=logits[j].topk(min(W,10))
    for v,d in zip(vals.tolist(),inds.tolist()): candidates.append((seqs[j]+[d],score+v))
   beams=sorted(candidates,key=lambda q:q[1],reverse=True)[:W]
  outputs=[]
  for seq,score in beams:
   ds=seq[2::3]; value=0
   for d in reversed(ds): value=value*10+d
   outputs.append(value)
  good += (a+b in outputs)
  if (z+1)%200==0: print(z+1,good/(z+1),flush=True)
