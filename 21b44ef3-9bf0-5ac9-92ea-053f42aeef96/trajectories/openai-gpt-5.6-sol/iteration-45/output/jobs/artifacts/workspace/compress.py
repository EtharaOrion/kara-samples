import sys
from pathlib import Path
import torch
sys.path.insert(0,'/workspace')
import submission
from torch import nn
class Legacy(nn.Module):
 def __init__(self):
  super().__init__(); D=20
  self.token=nn.Embedding(11,D); self.position=nn.Parameter(torch.empty(25,D))
  self.norm_attn=nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
  self.query=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
  self.key=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
  self.value=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
  self.proj=nn.ModuleList([nn.Linear(D,D,bias=False) for _ in range(2)])
  self.norm_ff=nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
  self.ff1=nn.ModuleList([nn.Linear(D,4,bias=False) for _ in range(2)])
  self.ff2=nn.ModuleList([nn.Linear(4,D,bias=False) for _ in range(2)])
  self.final_norm=nn.LayerNorm(D); self.head=nn.Linear(D,10,bias=False)
old=Legacy()
old.load_state_dict(torch.load('/workspace/final.pt',map_location='cpu',weights_only=True))
sd=old.state_dict()
arrays=[]
# Registration order through proj.
arrays += [sd['position'],sd['token.weight']]
for i in range(2): arrays += [sd[f'norm_attn.{i}.weight'],sd[f'norm_attn.{i}.bias']]
for group in ('query','key','value','proj'):
    for i in range(2): arrays += [sd[f'{group}.{i}.weight']]
# non-affine norm_ff: fold into newly biased ff1
for i in range(2):
    w=sd[f'ff1.{i}.weight']; g=sd[f'norm_ff.{i}.weight']; beta=sd[f'norm_ff.{i}.bias']
    arrays += [w*g[None,:], w@beta]
for i in range(2): arrays += [sd[f'ff2.{i}.weight']]
# non-affine final norm and class-9 reference gauge
w=sd['head.weight']; g=sd['final_norm.weight']; beta=sd['final_norm.bias']
d=w[:9]-w[9:10]
arrays += [d*g[None,:], d@beta]

source=Path('/workspace/submission.py').read_text().split('_TRAINED_STATE = [',1)[0]
source=source.replace('self.norm_ff = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])','self.norm_ff = nn.ModuleList([nn.LayerNorm(D, elementwise_affine=False) for _ in range(2)])')
source=source.replace('self.ff1 = nn.ModuleList([nn.Linear(D, FF, bias=False) for _ in range(2)])','self.ff1 = nn.ModuleList([nn.Linear(D, FF, bias=True) for _ in range(2)])')
source=source.replace('self.final_norm = nn.LayerNorm(D)','self.final_norm = nn.LayerNorm(D, elementwise_affine=False)')
source=source.replace('self.head = nn.Linear(D, 10, bias=False)','self.head = nn.Linear(D, 9, bias=True)')
source=source.replace('        return self.head(self.final_norm(x))','        logits = self.head(self.final_norm(x))\n        return torch.cat((logits, torch.zeros_like(logits[..., :1])), dim=-1)')
vals=[]
for p in arrays:
    vals.append('['+','.join(format(x,'.9g') for x in p.float().reshape(-1).tolist())+']')
source += '_TRAINED_STATE = [\n'+',\n'.join(vals)+'\n]\n\n\n'
# append public functions from original
original=Path('/workspace/submission.py').read_text()
source += 'def build_model():'+original.split('def build_model():',1)[1]
Path('/workspace/submission.py').write_text(source)
print('array params',sum(x.numel() for x in arrays))
