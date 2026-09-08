import argparse, torch
import train
p=argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('label'); p.add_argument('width',type=int); p.add_argument('--steps',type=int,default=12000); p.add_argument('--lr',type=float,default=2e-5); a=p.parse_args()
ck=torch.load('/workspace/'+a.src,weights_only=True)
m=train.AdditionTransformer(ck['width']).cuda(); m.load_state_dict(ck['state']); m=train.prune(m,a.width)
train.train_stage(m,a.steps,a.lr,8192,.5,a.label)
print('R',train.validate(m,200000,False)); print('S',train.validate(m,200000,True))
