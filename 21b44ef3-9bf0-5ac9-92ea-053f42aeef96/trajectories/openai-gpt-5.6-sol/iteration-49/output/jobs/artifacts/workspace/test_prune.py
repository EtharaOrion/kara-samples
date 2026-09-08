import sys,torch,time
sys.path.insert(0,'/workspace');import submission
from train import batch
m,_=submission.build_model();m.cuda().eval()
flat=m.tok.view(-1); order=flat.abs().argsort()
@torch.no_grad()
def acc(n,st):
 good=0;margin=100
 for off in range(0,n,20000):
  inp,y=batch(min(20000,n-off),st);seq=inp[:,:17];out=[]
  for j in range(9):
   log=m(seq)[:,-1];q=log.topk(2,-1).values;margin=min(margin,float((q[:,0]-q[:,1]).min()));d=log.argmax(-1);out.append(d)
   if j<8:seq=torch.cat((seq,d[:,None]),1)
  good+=(torch.stack(out,1)==y).all(1).sum().item()
 return good,margin
for nzero in [8,16,24,32,48,64]:
 with torch.no_grad():flat[order[:nzero]]=0
 print(nzero,'maxpruned',float(flat.new_tensor([m.tok.view(-1)[i] for i in order[:nzero]]).abs().max()),acc(300000,0),acc(300000,.7),flush=True)
