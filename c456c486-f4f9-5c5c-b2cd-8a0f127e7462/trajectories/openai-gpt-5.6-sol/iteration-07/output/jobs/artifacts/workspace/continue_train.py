import argparse, json, time, torch
import torch.nn.functional as F
from train import Adder, batch_data, greedy
ap=argparse.ArgumentParser(); ap.add_argument('--infile',required=True); ap.add_argument('--out',required=True); ap.add_argument('--steps',type=int,default=5000); ap.add_argument('--batch',type=int,default=4096); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--structured',type=float,default=0.0); args=ap.parse_args()
ck=torch.load(args.infile,weights_only=True); model=Adder(ck['sharing']).cuda(); model.load_state_dict(ck['state']); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.002); t=time.time()
for step in range(1,args.steps+1):
    seq,y=batch_data(args.batch,'cuda',args.structured); logits=model(seq); loss=F.cross_entropy(logits[:,1::3].reshape(-1,10),y.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step%250==0: print(json.dumps({'step':step,'loss':loss.item(),'digit':(logits[:,1::3].argmax(-1)==y).float().mean().item(),'sec':time.time()-t}),flush=True)
acc=greedy(model,10000); print(json.dumps({'accuracy':acc}),flush=True); torch.save({'state':model.state_dict(),'sharing':ck['sharing'],'parameters':sum(p.numel() for p in model.parameters()),'accuracy':acc},args.out)
