import sys, torch
sys.path.insert(0,'/workspace')
from train import AddTransformer, accuracy
D=20;H=4;HD=5
base=AddTransformer(2).cuda(); base.load_state_dict(torch.load('/workspace/final.pt',weights_only=True)['model']);base.eval()
s={k:v.detach().clone() for k,v in base.state_dict().items()}
# Fold affine FFN LN into its linear map.
s['f1.bias']=s['f1.weight']@s['fn.bias'];s['f1.weight']=s['f1.weight']*s['fn.weight'].unsqueeze(0)
# Fold final affine LN into classifier.
s['head.bias']=s['head.weight']@s['outn.bias'];s['head.weight']=s['head.weight']*s['outn.weight'].unsqueeze(0)
# Reference class 9 logits (argmax invariant).
s['head.bias']=s['head.bias'][:9]-s['head.bias'][9];s['head.weight']=s['head.weight'][:9]-s['head.weight'][9]
# Null columns for non-affine layer-normalized inputs.
s['f1.weight']=s['f1.weight']-s['f1.weight'][:,-1,None]
s['head.weight']=s['head.weight']-s['head.weight'][:,-1,None]
# Residual all-ones gauges: force final residual coordinate to zero for every source.
s['tok.weight']=s['tok.weight']-s['tok.weight'][:,-1,None]
s['pb']=s['pb']-s['pb'][:,-1,None]
for z in range(2): s[f'o.{z}.weight']=s[f'o.{z}.weight']-s[f'o.{z}.weight'][-1:,:]
s['f2.weight']=s['f2.weight']-s['f2.weight'][-1:,:]
# Positional GL(2): choose best-conditioned pair, make those rows the identity.
best=None
for i in range(25):
 for j in range(i+1,25):
  A=s['pa'][[i,j]]; score=abs(torch.det(A)).item()
  if best is None or score>best[0]:best=(score,i,j,A)
_,pi,pj,A=best
s['pa']=s['pa']@torch.linalg.inv(A);s['pb']=A@s['pb']
# Independent K and V GL(5) gauges, selecting a stable set of five input columns.
def basis_gauge(name):
 W=s[name+'.weight']
 # QR pivoting unavailable: greedily grow volume by testing combinations.
 chosen=[]
 for _ in range(5):
  cand=[]
  for c in range(20):
   if c in chosen:continue
   M=W[:,chosen+[c]]
   cand.append((torch.linalg.svdvals(M)[-1].item(),c))
  chosen.append(max(cand)[1])
 A=W[:,chosen]; inv=torch.linalg.inv(A);s[name+'.weight']=inv@W
 if name=='k':
  for z in range(2):
   Q=s[f'q.{z}.weight'];s[f'q.{z}.weight']=torch.cat([A.T@Q[h*5:(h+1)*5] for h in range(H)],0)
 else:
  for z in range(2):
   O=s[f'o.{z}.weight']; blocks=[]
   for h in range(H):blocks.append(O[:,h*5:(h+1)*5]@A)
   s[f'o.{z}.weight']=torch.cat(blocks,1)
 return chosen
kc=basis_gauge('k');vc=basis_gauge('v')
print('gauges pos',pi,pj,'k',kc,'v',vc)
# Re-apply residual output gauge after V transform.
for z in range(2):s[f'o.{z}.weight']=s[f'o.{z}.weight']-s[f'o.{z}.weight'][-1:,:]

class C(torch.nn.Module):
 def __init__(self):
  super().__init__()
  P=torch.nn.Parameter
  self.tok=P(s['tok.weight'][:,:19]); self.pa=P(torch.cat((s['pa'][:pi],s['pa'][pi+1:pj],s['pa'][pj+1:])),requires_grad=False);self.pb=P(s['pb'][:,:19])
  self.q=torch.nn.ParameterList([P(s[f'q.{z}.weight']) for z in range(2)])
  self.o=torch.nn.ParameterList([P(s[f'o.{z}.weight'][:19]) for z in range(2)])
  self.k=P(s['k.weight'][:,[c for c in range(20) if c not in kc]]);self.v=P(s['v.weight'][:,[c for c in range(20) if c not in vc]])
  self.aw=P(s['an.weight']);self.ab=P(s['an.bias']);self.f1=P(s['f1.weight'][:,:19]);self.fb=P(s['f1.bias']);self.f2=P(s['f2.weight'][:19]);self.hw=P(s['head.weight'][:,:19]);self.hb=P(s['head.bias'])
 def full(self):pass

def exporter(path='/workspace/submission.py'):
 c=C().cpu(); vals={k:v.detach().tolist() for k,v in c.named_parameters()}
 src='''import torch\nfrom torch import nn\nimport torch.nn.functional as F\nD=20;H=4;HD=5; PI='''+str(pi)+''';PJ='''+str(pj)+''';KC='''+repr(kc)+''';VC='''+repr(vc)+'''\nclass Model(nn.Module):\n def __init__(self):\n  super().__init__(); P=nn.Parameter\n'''
 for k,v in vals.items():src+='  self.'+k.replace('.', '_')+'=P(torch.tensor('+repr(v)+'))\n'
 src+=''' def forward(self,t):\n  L=t.shape[1]; zero=self.tok.new_zeros(*self.tok.shape[:-1],1); tok=torch.cat((self.tok,zero),1); pb=torch.cat((self.pb,self.pb.new_zeros(2,1)),1); fixed=self.pa.new_tensor([[1.,0.],[0.,1.]]); pa=torch.cat((self.pa[:PI],fixed[:1],self.pa[PI:PJ-1],fixed[1:],self.pa[PJ-1:]),0); x=tok[t]+pa[:L]@pb; mask=torch.triu(torch.ones(L,L,device=t.device,dtype=torch.bool),1)\n  cols=[i for i in range(D) if i not in KC]; kfull=self.k.new_zeros(HD,D); kfull[:,KC]=torch.eye(HD,device=t.device);kfull[:,cols]=self.k; cols=[i for i in range(D) if i not in VC];vfull=self.v.new_zeros(HD,D);vfull[:,VC]=torch.eye(HD,device=t.device);vfull[:,cols]=self.v\n  for z in range(2):\n   y=F.layer_norm(x,(D,),self.aw,self.ab); q=F.linear(y,getattr(self,'q_'+str(z))).view(-1,L,H,HD).transpose(1,2); k=F.linear(y,kfull).unsqueeze(1);v=F.linear(y,vfull).unsqueeze(1);w=torch.softmax((q@k.transpose(-2,-1))/(HD**.5)+mask.to(x.dtype)*-1e4,dim=-1); oi=getattr(self,'o_'+str(z));ofull=torch.cat((oi,oi.new_zeros(1,D)),0);x=x+F.linear((w@v).transpose(1,2).reshape(-1,L,D),ofull);n=F.layer_norm(x,(D,),None,None);fw=torch.cat((self.f1,self.f1.new_zeros(2,1)),1);f2=torch.cat((self.f2,self.f2.new_zeros(1,2)),0);x=x+F.linear(F.gelu(F.linear(n,fw,self.fb)),f2)\n  n=F.layer_norm(x,(D,),None,None);hw=torch.cat((self.hw,self.hw.new_zeros(9,1)),1);log=F.linear(n,hw,self.hb);return torch.cat((log,log.new_zeros(*log.shape[:-1],1)),-1)\ndef build_model():\n m=Model();m.eval();return m,{'architecture':'trained causal grouped-query digit transformer','training':'AdamW on generated full-width addition pairs','parameters':sum(p.numel() for p in m.parameters())}\ndef add(model,a:int,b:int)->int:\n da=str(a)[::-1];db=str(b)[::-1];tokens=[]\n for x,y in zip(da,db):tokens.extend((int(x),int(y)))\n seq=torch.tensor([tokens+[10]],device=next(model.parameters()).device);out=[]\n with torch.no_grad():\n  for i in range(9):\n   d=int(model(seq)[0,-1].argmax());out.append(d)\n   if i<8:seq=torch.cat((seq,seq.new_tensor([[d]])),1)\n return int(''.join(str(x) for x in out[::-1]))\n'''
 open(path,'w').write(src);print('params',sum(p.numel() for p in c.parameters()))
exporter()
