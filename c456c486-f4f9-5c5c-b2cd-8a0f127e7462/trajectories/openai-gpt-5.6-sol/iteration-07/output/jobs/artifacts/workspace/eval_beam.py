import argparse, torch
from train import Adder, digits, LIMIT, DIGITS
@torch.inference_mode()
def beam(model,count,width,batch=512,seed=987654321):
 g=torch.Generator().manual_seed(seed); aa=torch.randint(LIMIT,(count,),generator=g); bb=torch.randint(LIMIT,(count,),generator=g); good=0
 for st in range(0,count,batch):
  a=aa[st:st+batch].cuda(); b=bb[st:st+batch].cuda(); B=a.numel(); ad=digits(a); bd=digits(b); target=digits(a+b)
  seq=torch.zeros(B,width,45,dtype=torch.long,device='cuda'); seq[:,:,0::3]=ad[:,None]; seq[:,:,1::3]=bd[:,None]; scores=torch.full((B,width),-1e30,device='cuda'); scores[:,0]=0
  for i in range(15):
   logp=model(seq[:,:,:3*i+2].reshape(B*width,3*i+2))[:,-1].log_softmax(-1).reshape(B,width,10)
   vals,inds=(scores[:,:,None]+logp).reshape(B,-1).topk(width,-1); parent=inds//10; digit=inds%10
   seq=seq.gather(1,parent[:,:,None].expand(-1,-1,45)); seq[:,:,3*i+2]=digit; scores=vals
  pred=seq[:,0,2::3]; good+=(pred==target).all(1).sum().item()
 return good/count
ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--width',type=int,default=2); ap.add_argument('--count',type=int,default=10000); args=ap.parse_args(); ck=torch.load(args.file,weights_only=True); m=Adder(ck['sharing']).cuda().eval(); m.load_state_dict(ck['state']); print(beam(m,args.count,args.width))
