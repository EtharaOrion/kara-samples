import torch, torch.nn.functional as F
from train import Model,batch,evaluate,export
class Gated(Model):
 def __init__(self,ff,drop): super().__init__(ff); self.drop=drop; self.gate=1.
 def forward(self,tokens):
  n=tokens.shape[1]; x=self.token(tokens)+self.pos_left[:n]@self.pos_right
  mask=torch.ones(n,n,device=tokens.device,dtype=torch.bool).triu(1)
  for i in range(2):
   z=self.norm1(x); b=z.shape[0]; q=self.q[i](z).view(b,n,4,5).transpose(1,2); k,v=self.k(z),self.v(z)
   attn=torch.einsum('bhtd,bsd->bhts',q,k).mul_(5**-.5).masked_fill(mask,-torch.inf)
   c=torch.einsum('bhts,bsd->bhtd',attn.softmax(-1),v)
   x=x+self.o[i](c.transpose(1,2).reshape(b,n,20))
   h=F.gelu(self.ff1(self.norm2(x))); h[:,:,self.drop]*=self.gate
   x=x+self.ff2(h)
  return self.head(self.final_norm(x))
def anneal(source,drop,steps,lr):
 m=Gated(source.ff_width,drop).cuda(); m.load_state_dict(source.state_dict()); opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(.9,.98),weight_decay=.01)
 for step in range(steps):
  m.gate=max(0.,1-step/(steps*.7)); tok,tgt=batch(structured_fraction=.45); loss=F.cross_entropy(m(tok)[:,16:].reshape(-1,10),tgt.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
  if (step+1)%2000==0: print('anneal',source.ff_width,step+1,m.gate,float(loss),flush=True)
 m.gate=0.; keep=torch.tensor([i for i in range(source.ff_width) if i!=drop],device='cuda'); new=Model(source.ff_width-1).cuda(); sd=m.state_dict(); d=new.state_dict()
 for name in d:
  if name=='ff1.weight': d[name].copy_(sd[name][keep])
  elif name=='ff2.weight': d[name].copy_(sd[name][:,keep])
  else:d[name].copy_(sd[name])
 return new
m=Model(4).cuda(); m.load_state_dict(torch.load('/workspace/teacher_stable.pt',weights_only=True))
# Neuron 1 had best direct-pruning score.
m=anneal(m,1,14000,2e-5); print('w3',evaluate(m,200000,.5),flush=True); torch.save(m.state_dict(),'/workspace/anneal3.pt')
if evaluate(m,50000,.5)[0]>=49500:
 m=anneal(m,0,22000,1e-5); print('w2',evaluate(m,500000,.5),flush=True); torch.save(m.state_dict(),'/workspace/anneal2.pt'); export(m)
