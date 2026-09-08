import argparse, torch
import train
p=argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('label'); p.add_argument('--steps',type=int,default=15000); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--structured',type=float,default=.4); a=p.parse_args()
ck=torch.load('/workspace/'+a.src,weights_only=True)
m=train.AdditionTransformer(ck['width']).cuda(); m.load_state_dict(ck['state'])
train.train_stage(m,a.steps,a.lr,8192,a.structured,a.label)
print('R',train.validate(m,200000,False)); print('S',train.validate(m,200000,True)); train.export(m)
