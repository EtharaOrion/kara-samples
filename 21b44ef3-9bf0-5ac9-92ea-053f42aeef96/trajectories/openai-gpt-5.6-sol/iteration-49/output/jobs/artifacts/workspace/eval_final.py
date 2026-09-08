import sys,torch,time
sys.path.insert(0,'/workspace');import submission
from train import batch
m,_=submission.build_model();m.cuda().eval()
@torch.no_grad()
def run(n,structured):
 good=0; margin=100.;t=time.time()
 for off in range(0,n,10000):
  z=min(10000,n-off); inp,y=batch(z,structured);seq=inp[:,:17];pred=[]
  for j in range(9):
   log=m(seq)[:,-1]; top=log.topk(2,-1).values;margin=min(margin,float((top[:,0]-top[:,1]).min()));d=log.argmax(-1);pred.append(d)
   if j<8:seq=torch.cat((seq,d[:,None]),1)
  good+=(torch.stack(pred,1)==y).all(1).sum().item()
 print(structured,good,n,'margin',margin,'sec',time.time()-t,flush=True)
run(500000,0);run(500000,.7)
