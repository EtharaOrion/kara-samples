import torch, sys
sys.path.insert(0,'/workspace')
from torch import nn
class Base(nn.Module):
 def __init__(self):
  super().__init__(); self.token=nn.Parameter(torch.empty(11,20)); self.pos_a=nn.Parameter(torch.empty(25,2)); self.pos_b=nn.Parameter(torch.empty(2,20)); self.norm_att=nn.LayerNorm(20); self.norm_ff=nn.LayerNorm(20); self.q=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.k=nn.Linear(20,5,bias=False); self.v=nn.Linear(20,5,bias=False); self.o=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.ff1=nn.Linear(20,2,bias=False); self.ff2=nn.Linear(2,20,bias=False); self.final_norm=nn.LayerNorm(20); self.head=nn.Linear(20,10,bias=False)
m=Base(); m.load_state_dict(torch.load('/workspace/w2.pt',map_location='cpu',weights_only=True)); m.eval()
s=m.state_dict()
# Fold FF and final LayerNorm affine maps into their following linear maps.
ffw=s['ff1.weight']*s['norm_ff.weight'][None,:]
ffb=s['ff1.weight']@s['norm_ff.bias']
hw=s['head.weight']*s['final_norm.weight'][None,:]
hb=s['head.weight']@s['final_norm.bias']
# Center positional matrix at separator, absorb offset into every token embedding.
P=s['pos_a']@s['pos_b']; anchor=16; shift=P[anchor].clone(); P=P-shift; token=s['token']+shift
# Choose a well-conditioned pair of positional rows as fixed basis.
best=None
for i in range(25):
 for j in range(i+1,25):
  if i==anchor or j==anchor: continue
  d=abs(torch.linalg.det(P[[i,j]]@P[[i,j]].T)).item()
  if best is None or d>best[0]: best=(d,i,j)
i,j=best[1:]; basis=P[[i,j]].clone(); coeff=P@torch.linalg.pinv(basis)
free=[r for r in range(25) if r not in (anchor,i,j)]
# Independent K and V head-space gauges, fixing first five input columns to identity.
K=s['k.weight']; A=torch.linalg.inv(K[:,:5]); Knew=A@K
V=s['v.weight']; B=torch.linalg.inv(V[:,:5]); Vnew=B@V
q=[]; o=[]
for layer in range(2):
 W=s[f'q.{layer}.weight']; q.append(torch.cat([torch.linalg.solve(A.T,W[h*5:(h+1)*5]) for h in range(4)],0))
 W=s[f'o.{layer}.weight']; o.append(torch.cat([W[:,h*5:(h+1)*5]@torch.linalg.inv(B) for h in range(4)],1))
# LayerNorm discards the all-ones component of every token and positional vector.
token = token - token.mean(1, keepdim=True)
basis = basis - basis.mean(1, keepdim=True)
hw=hw[:9]-hw[9]; hb=hb[:9]-hb[9]
state=[token[:,:19], coeff[free], basis[:,:19], s['norm_att.weight'],s['norm_att.bias'],q[0],q[1],Knew[:,5:],Vnew[:,5:],o[0],o[1],ffw,ffb,s['ff2.weight'],hw,hb]
torch.save({'state':state,'fixed_rows':(anchor,i,j),'free':free},'/workspace/compressed.pt')
print('fixed rows',anchor,i,j,'params',sum(x.numel() for x in state))
# Write final inference file with direct source-level tensor literals.
def literal(t): return repr(t.detach().reshape(-1).tolist())
text='''import torch\nfrom torch import nn\nimport torch.nn.functional as F\n\nclass AdditionTransformer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        shapes = [(11,19),(22,2),(2,19),(20,),(20,),(20,20),(20,20),(5,15),(5,15),(20,20),(20,20),(2,20),(2,),(20,2),(9,20),(9,)]\n        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(v).reshape(s)) for v,s in zip(_WEIGHTS,shapes)])\n\n    def forward(self, tokens):\n        tokenfree,pa,pbfree,nw,nb,q0,q1,kfree,vfree,o0,o1,f1,fb,f2,head,hb = self.weights\n        n=tokens.shape[1]\n        token=torch.cat((tokenfree,-tokenfree.sum(1,keepdim=True)),1)\n        pb=torch.cat((pbfree,-pbfree.sum(1,keepdim=True)),1)\n        pos=torch.empty(25,2,device=tokens.device,dtype=token.dtype)\n        pos[_FREE]=pa; pos[16]=0; pos[_R0]=torch.tensor([1.,0.],device=tokens.device); pos[_R1]=torch.tensor([0.,1.],device=tokens.device)\n        x=F.embedding(tokens,token)+(pos@pb)[:n]\n        mask=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()\n        eye=torch.eye(5,device=x.device,dtype=x.dtype)\n        kweight=torch.cat((eye,kfree),1); vweight=torch.cat((eye,vfree),1)\n        for qweight,oweight in ((q0,o0),(q1,o1)):\n            z=F.layer_norm(x,(20,),nw,nb)\n            q=F.linear(z,qweight).view(-1,n,4,5).transpose(1,2)\n            k=F.linear(z,kweight)[:,None]; v=F.linear(z,vweight)[:,None]\n            att=((q@k.transpose(-2,-1))*(5.**-.5)).masked_fill(~mask,-torch.inf).softmax(-1)\n            x=x+F.linear((att@v).transpose(1,2).reshape(-1,n,20),oweight)\n            z=F.layer_norm(x,(20,),None,None)\n            x=x+F.linear(F.gelu(F.linear(z,f1,fb)),f2)\n        logits=F.linear(F.layer_norm(x,(20,),None,None),head,hb)
        return torch.cat((logits,torch.zeros_like(logits[...,:1])), -1)\n\ndef build_model():\n    model=AdditionTransformer(); model.eval()\n    return model,{"architecture":"causal grouped-query digit transformer","digits":8}\n\ndef add(model,a:int,b:int)->int:\n    sa,sb=f"{a:08d}"[::-1],f"{b:08d}"[::-1]\n    tokens=[int(x) for pair in zip(sa,sb) for x in pair]+[10]\n    device=next(model.parameters()).device\n    with torch.no_grad():\n        for _ in range(9):\n            x=torch.tensor([tokens],dtype=torch.long,device=device)\n            tokens.append(int(model(x)[0,-1].argmax()))\n    return int("".join(str(x) for x in tokens[-9:][::-1]))\n'''
prefix='_FREE='+repr(free)+'\n_R0='+str(i)+'\n_R1='+str(j)+'\n_WEIGHTS=[\n'+',\n'.join(literal(x) for x in state)+'\n]\n'
open('/workspace/submission.py','w').write('import torch\n'+prefix+text.replace('import torch\n','',1))
