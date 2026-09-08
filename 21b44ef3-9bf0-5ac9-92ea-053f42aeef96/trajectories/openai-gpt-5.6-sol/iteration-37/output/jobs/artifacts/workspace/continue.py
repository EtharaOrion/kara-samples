import sys, torch, random
sys.path.insert(0,'/workspace')
import train

torch.manual_seed(2026); random.seed(2026); torch.backends.cuda.matmul.allow_tf32=True
m=train.AdditionTransformer(4).cuda(); m.load_state_dict(torch.load('/workspace/teacher.pt',weights_only=True))
m=train.prune(m,3)
train.train_stage(m,18000,2e-5,.42,'width3')
print('width3 final',train.exact_accuracy(m,300000,0),train.exact_accuracy(m,300000,.7),flush=True)
train.export(m)
m=train.prune(m,2)
train.train_stage(m,30000,1.2e-5,.45,'width2a')
train.train_stage(m,16000,4e-6,.40,'width2b')
print('width2 final',train.exact_accuracy(m,1000000,0),train.exact_accuracy(m,1000000,.7),flush=True)
torch.save(m.state_dict(),'/workspace/final.pt'); train.export(m)
