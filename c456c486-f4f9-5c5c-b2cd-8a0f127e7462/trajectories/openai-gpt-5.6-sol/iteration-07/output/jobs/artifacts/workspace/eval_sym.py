import argparse, torch
from train import Adder, digits, LIMIT
@torch.inference_mode()
def evaluate(model,count,batch=512,seed=987654321):
 g=torch.Generator().manual_seed(seed); AA=torch.randint(LIMIT,(count,),generator=g); BB=torch.randint(LIMIT,(count,),generator=g); good=0
 for st in range(0,count,batch):
  aa=AA[st:st+batch].cuda(); bb=BB[st:st+batch].cuda(); B=aa.numel(); ads=[digits(aa),digits(bb)]; bds=[digits(bb),digits(aa)]; candidates=[]
  for order in range(2):
   seq=torch.zeros(B,45,dtype=torch.long,device='cuda'); seq[:,0::3]=ads[order]; seq[:,1::3]=bds[order]
   for i in range(15): seq[:,3*i+2]=model(seq[:,:3*i+2])[:,-1].argmax(-1)
   candidates.append(seq[:,2::3].clone())
  score=torch.zeros(B,2,device='cuda')
  for ci in range(2):
   for order in range(2):
    seq=torch.zeros(B,45,dtype=torch.long,device='cuda'); seq[:,0::3]=ads[order]; seq[:,1::3]=bds[order]; seq[:,2::3]=candidates[ci]
    lp=model(seq)[:,1::3].log_softmax(-1); score[:,ci].add_(lp.gather(-1,candidates[ci][...,None]).squeeze(-1).sum(-1))
  choice=score.argmax(-1); pred=torch.stack(candidates,1)[torch.arange(B,device='cuda'),choice]; target=digits(aa+bb); good+=(pred==target).all(1).sum().item()
 return good/count
ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--count',type=int,default=10000); args=ap.parse_args(); ck=torch.load(args.file,weights_only=True); m=Adder(ck['sharing']).cuda().eval(); m.load_state_dict(ck['state']); print(evaluate(m,args.count))
