from pathlib import Path
import itertools
import torch
from train import Model, D, H, DH, N

m=Model(2)
m.load_state_dict(torch.load('/workspace/final.pt',map_location='cpu',weights_only=True))
s={k:v.detach().float().clone() for k,v in m.state_dict().items()}

# Fold the two affine normalizations omitted by inference.
fw=s['ff1.weight']*s['ff_norm.weight'][None,:]
fb=s['ff1.bias']+s['ff1.weight']@s['ff_norm.bias']
cw=s['classifier.weight']*s['final_norm.weight'][None,:]
cb=s['classifier.bias']+s['classifier.weight']@s['final_norm.bias']

# Input row translations are invisible to every subsequent LayerNorm.
tok=s['token.weight']-s['token.weight'].mean(1,keepdim=True)
tok_free=tok[:,:19]
b=s['pos_b']-s['pos_b'].mean(1,keepdim=True)
a=s['pos_a']
# Exact GL(2) gauge; choose a well-conditioned pair and fix those rows to I.
best=max(itertools.combinations(range(N),2),key=lambda ij:abs(float(torch.linalg.det(a[list(ij)]))))
sel=a[list(best)]
a2=a@torch.linalg.inv(sel)
b2=sel@b
free=torch.tensor([i for i in range(N) if i not in best])
afree=a2[free]

# Independent GL(5) gauges fix five rows of K and V to identity.
def kvfix(mat):
    best_rows=max(itertools.combinations(range(D),DH),key=lambda ij:abs(float(torch.linalg.det(mat[list(ij)]))))
    S=mat[list(best_rows)]
    out=mat@torch.linalg.inv(S)
    free_rows=torch.tensor([i for i in range(D) if i not in best_rows])
    return out[free_rows],best_rows,S
kfree,kfix,ks=kvfix(s['k'])
vfree,vfix,vs=kvfix(s['v'])
q=[]; o=[]
for layer in range(2):
    qq=s[f'q.{layer}'].reshape(D,H,DH)
    q.append(torch.einsum('dhj,jk->dhk',qq,ks.T).reshape(D,D))
    oo=s[f'o.{layer}'].reshape(H,DH,D)
    o.append(torch.einsum('ij,hjd->hid',vs,oo).reshape(D,D))

# Class 9 is the zero reference logit; softmax/argmax are translation invariant.
cw=cw-cw[9:10]; cb=cb-cb[9]
state={
 'token_free':tok_free,'pos_free':afree,'pos_b_free':b2[:,:19],
 'q.0':q[0],'q.1':q[1],'o.0':o[0],'o.1':o[1],
 'k_free':kfree,'v_free':vfree,
 'attn_weight':s['attn_norm.weight'],'attn_bias':s['attn_norm.bias'],
 'ff1_free':(fw-fw.mean(1,keepdim=True))[:,:19],'ff1_bias':fb,'ff2.weight':s['ff2.weight'],
 'classifier_free':(cw[:9]-cw[:9].mean(1,keepdim=True))[:,:19],'classifier_bias':cb[:9],
}

def lit(t):return 'torch.tensor('+repr(t.tolist())+',dtype=torch.float32)'
weights='_STATE={\n'+''.join(' '+repr(k)+':'+lit(v)+',\n' for k,v in state.items())+'}\n'
template='''import torch
from torch import nn
import torch.nn.functional as F
D,H,DH,N=20,4,5,25
POS_FIXED=%r
K_FIXED=%r
V_FIXED=%r
class AdditionTransformer(nn.Module):
 def __init__(self):
  super().__init__()
  self.token_free=nn.Parameter(torch.empty(11,19)); self.pos_free=nn.Parameter(torch.empty(23,2)); self.pos_b_free=nn.Parameter(torch.empty(2,19))
  self.q=nn.ParameterList([nn.Parameter(torch.empty(D,D)) for _ in range(2)]); self.o=nn.ParameterList([nn.Parameter(torch.empty(D,D)) for _ in range(2)])
  self.k_free=nn.Parameter(torch.empty(15,DH)); self.v_free=nn.Parameter(torch.empty(15,DH)); self.attn_weight=nn.Parameter(torch.empty(D)); self.attn_bias=nn.Parameter(torch.empty(D))
  self.ff1_free=nn.Parameter(torch.empty(2,19)); self.ff1_bias=nn.Parameter(torch.empty(2)); self.ff2=nn.Linear(2,D,bias=False); self.classifier_free=nn.Parameter(torch.empty(9,19)); self.classifier_bias=nn.Parameter(torch.empty(9))
  self.register_buffer("mask",torch.triu(torch.ones(N,N,dtype=torch.bool),1),persistent=False); self.load_state_dict(_STATE); self.eval()
 def _rows(self,free,fixed):
  z=free.new_zeros(D,DH); z[list(fixed)]=torch.eye(DH,device=free.device,dtype=free.dtype); keep=[i for i in range(D) if i not in fixed]; z[keep]=free; return z
 def forward(self,t):
  L=t.shape[1]; token=torch.cat((self.token_free,-self.token_free.sum(1,keepdim=True)),1)
  pb=torch.cat((self.pos_b_free,-self.pos_b_free.sum(1,keepdim=True)),1); pa=self.pos_free.new_zeros(N,2); pa[list(POS_FIXED)]=torch.eye(2,device=t.device,dtype=pb.dtype); keep=[i for i in range(N) if i not in POS_FIXED]; pa[keep]=self.pos_free
  x=F.embedding(t,token)+(pa[:L]@pb); k=self._rows(self.k_free,K_FIXED); v=self._rows(self.v_free,V_FIXED)
  for i in range(2):
   z=F.layer_norm(x,(D,),self.attn_weight,self.attn_bias); q=(z@self.q[i]).view(-1,L,H,DH).transpose(1,2); kk=(z@k).unsqueeze(1); vv=(z@v).unsqueeze(1)
   scores=(q@kk.transpose(-1,-2))*(DH**-.5); scores=scores.masked_fill(self.mask[:L,:L],-1e4); x=x+(scores.softmax(-1)@vv).transpose(1,2).reshape(-1,L,D)@self.o[i]
   fn=F.layer_norm(x,(D,)); fw=torch.cat((self.ff1_free,-self.ff1_free.sum(1,keepdim=True)),1); x=x+self.ff2(F.gelu(F.linear(fn,fw,self.ff1_bias)))
  zn=F.layer_norm(x,(D,)); cw=torch.cat((self.classifier_free,-self.classifier_free.sum(1,keepdim=True)),1); nine=F.linear(zn,cw,self.classifier_bias); return torch.cat((nine,torch.zeros_like(nine[...,:1])),2)
def build_model():
 m=AdditionTransformer(); return m,{"architecture":"causal grouped-query digit transformer","digit_order":"least-significant-first"}
def add(model,a:int,b:int)->int:
 sa=f"{a:08d}"[::-1]; sb=f"{b:08d}"[::-1]; tokens=[]
 for da,db in zip(sa,sb): tokens.extend((ord(da)-48,ord(db)-48))
 tokens.append(10); device=next(model.parameters()).device; seq=torch.tensor([tokens],dtype=torch.long,device=device); out=[]
 with torch.no_grad():
  for _ in range(9):
   d=int(model(seq)[0,-1].argmax().item()); out.append(chr(48+d)); seq=torch.cat((seq,torch.tensor([[d]],device=device)),1)
 return int("".join(reversed(out)))
'''%(best,kfix,vfix)
Path('/workspace/submission.py').write_text(template+'\n'+weights)
print('fixed',best,kfix,vfix,'parameters',sum(x.numel() for x in state.values()))
