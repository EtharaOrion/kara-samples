import sys
sys.path.insert(0,'/workspace')
import torch
ck=torch.load('/workspace/full_final.pt',map_location='cpu',weights_only=True)['state']
out={}
out['token.weight']=ck['token.weight']
out['pos']=ck['pos']
for group in ['q','k','v','o']:
 for i in range(2): out[f'{group}.{i}.weight']=ck[f'{group}.{i}.weight']
for i in range(2):
 out[f'an.{i}.weight']=ck[f'an.{i}.weight']; out[f'an.{i}.bias']=ck[f'an.{i}.bias']
 # Fold affine pre-FFN norm into the following projection.
 W=ck[f'fi.{i}.weight']; g=ck[f'fn.{i}.weight']; b=ck[f'fn.{i}.bias']
 out[f'fi.{i}.weight']=W*g[None,:]; out[f'fi.{i}.bias']=W@b
 out[f'fo.{i}.weight']=ck[f'fo.{i}.weight']
# Fold affine final norm into classifier.
W=ck['classifier.weight']; g=ck['final.weight']; b=ck['final.bias']
CW=W*g[None,:]; CB=W@b
out['classifier.weight']=CW[:9]-CW[9:10]; out['classifier.bias']=CB[:9]-CB[9]

def lit(t): return 'torch.tensor('+repr(t.tolist())+')'
state='{' + ',\n'.join(repr(k)+':'+lit(v) for k,v in out.items()) + '}'
template='''import torch
from torch import nn
import torch.nn.functional as F

class AdditionTransformer(nn.Module):
 def __init__(self):
  super().__init__()
  self.token=nn.Embedding(11,20); self.pos=nn.Parameter(torch.empty(25,20))
  self.q=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)])
  self.k=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)])
  self.v=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)])
  self.o=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)])
  self.an=nn.ModuleList([nn.LayerNorm(20) for _ in range(2)])
  self.fi=nn.ModuleList([nn.Linear(20,4) for _ in range(2)])
  self.fo=nn.ModuleList([nn.Linear(4,20,bias=False) for _ in range(2)])
  self.classifier=nn.Linear(20,9)
 def forward(self,tokens):
  batch,length=tokens.shape; x=self.token(tokens)+self.pos[:length]
  mask=torch.ones(length,length,dtype=torch.bool,device=tokens.device).tril()
  for i in range(2):
   z=self.an[i](x)
   q=self.q[i](z).view(batch,length,4,5).transpose(1,2)
   k=self.k[i](z).view(batch,length,4,5).transpose(1,2)
   v=self.v[i](z).view(batch,length,4,5).transpose(1,2)
   scores=(q@k.transpose(-2,-1))*(5.0**-0.5)
   scores=scores.masked_fill(~mask,torch.finfo(scores.dtype).min)
   attended=(scores.softmax(-1)@v).transpose(1,2).reshape(batch,length,20)
   x=x+self.o[i](attended)
   centered=(x-x.mean(-1,keepdim=True))*torch.rsqrt(x.var(-1,unbiased=False,keepdim=True)+1e-5)
   x=x+self.fo[i](F.gelu(self.fi[i](centered)))
  centered=(x-x.mean(-1,keepdim=True))*torch.rsqrt(x.var(-1,unbiased=False,keepdim=True)+1e-5)
  logits=self.classifier(centered)
  return torch.cat((logits,torch.zeros_like(logits[...,:1])),dim=-1)

_TRAINED_STATE=WEIGHT_PLACEHOLDER

def build_model():
 model=AdditionTransformer(); model.load_state_dict(_TRAINED_STATE); model.eval()
 return model,{"architecture":"two-layer causal digit transformer","digit_order":"least-significant-first"}

def add(model,a:int,b:int)->int:
 device=next(model.parameters()).device
 left,right=a,b; digits=[]
 for _ in range(8):
  digits.extend((left%10,right%10)); left//=10; right//=10
 tokens=torch.tensor([digits+[10]],dtype=torch.long,device=device); output=[]
 with torch.no_grad():
  for _ in range(9):
   digit=int(model(tokens)[0,-1].argmax()); output.append(digit)
   tokens=torch.cat((tokens,torch.tensor([[digit]],device=device)),1)
 result=0
 for digit in reversed(output): result=result*10+digit
 return result
'''.replace('WEIGHT_PLACEHOLDER',state)
open('/workspace/submission.py','w').write(template)
print('wrote',len(template),'bytes',sum(x.numel() for x in out.values()))
