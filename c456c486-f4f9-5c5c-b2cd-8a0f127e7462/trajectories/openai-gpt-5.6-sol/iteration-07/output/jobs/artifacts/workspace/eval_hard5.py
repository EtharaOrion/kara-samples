import argparse, torch
from train import Adder, digits, LIMIT
@torch.inference_mode()
def evaluate(model,count,batch=256,seed=246813579,W=5):
 g=torch.Generator().manual_seed(seed); AA=torch.randint(LIMIT,(count,),generator=g); BB=torch.randint(LIMIT,(count,),generator=g); good=0
 for st in range(0,count,batch):
  aa=AA[st:st+batch].cuda(); bb=BB[st:st+batch].cuda(); B=aa.numel(); ads=[digits(aa),digits(bb)]; bds=[digits(bb),digits(aa)]; candidates=[]
  for order in range(2):
   seq=torch.zeros(B,W,45,dtype=torch.long,device='cuda'); seq[:,:,0::3]=ads[order][:,None]; seq[:,:,1::3]=bds[order][:,None]; scores=torch.full((B,W),-1e30,device='cuda'); scores[:,0]=0
   for i in range(15):
    lp=model(seq[:,:,:3*i+2].reshape(B*W,3*i+2))[:,-1].log_softmax(-1).reshape(B,W,10); vals,inds=(scores[:,:,None]+lp).reshape(B,-1).topk(W,-1); par=inds//10; seq=seq.gather(1,par[:,:,None].expand(-1,-1,45)); seq[:,:,3*i+2]=inds%10; scores=vals
   candidates.extend(seq[:,:,2::3].unbind(1))
  C=len(candidates); score=torch.zeros(B,C,device='cuda')
  for ci in range(C):
   for order in range(2):
    seq=torch.zeros(B,45,dtype=torch.long,device='cuda'); seq[:,0::3]=ads[order]; seq[:,1::3]=bds[order]; seq[:,2::3]=candidates[ci]; lp=model(seq)[:,1::3].log_softmax(-1); score[:,ci].add_(lp.gather(-1,candidates[ci][...,None]).squeeze(-1).sum(-1))
  choice=score.argmax(-1); pred=torch.stack(candidates,1)[torch.arange(B,device='cuda'),choice]; good+=(pred==digits(aa+bb)).all(1).sum().item()
 return good/count
ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--count',type=int,default=2000); args=ap.parse_args(); ck=torch.load(args.file,weights_only=True); m=Adder(ck['sharing']).cuda().eval(); m.load_state_dict(ck['state']); print(evaluate(m,args.count))
