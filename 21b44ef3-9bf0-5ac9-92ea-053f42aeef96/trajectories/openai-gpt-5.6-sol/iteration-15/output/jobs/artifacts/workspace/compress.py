import math, torch
import torch.nn.functional as F
from torch import nn
from submission import AdditionTransformer
from train import batch, evaluate

class Factorized(AdditionTransformer):
    def __init__(self):
        super().__init__()
        del self.position
        self.position_a=nn.Parameter(torch.empty(25,7))
        self.position_b=nn.Parameter(torch.empty(7,20))
    def forward(self,tokens):
        length=tokens.shape[1]
        x=self.token(tokens)+(self.position_a @ self.position_b)[:length]
        mask=torch.triu(torch.ones(length,length,dtype=torch.bool,device=tokens.device),diagonal=1)
        for block in self.blocks: x=block(x,mask)
        return self.output(self.norm(x))

base=AdditionTransformer().cuda(); base.load_state_dict(torch.load('/workspace/best.pt',weights_only=True))
model=Factorized().cuda()
s={k:v for k,v in base.state_dict().items() if k!='position.weight'}
model.load_state_dict(s,strict=False)
u,sv,vh=torch.linalg.svd(base.position.weight.data,full_matrices=False)
model.position_a.data.copy_(u[:,:7]*sv[:7].sqrt()); model.position_b.data.copy_(sv[:7,None].sqrt()*vh[:7])
print('params',sum(p.numel() for p in model.parameters()),'initial',evaluate(model,50000,0),evaluate(model,50000,1),flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.001,fused=True)
for step in range(1,12001):
    sf=.5 if step<8000 else .7
    tokens,target,*_=batch(structured_fraction=sf)
    loss=F.cross_entropy(model(tokens)[:,16:25].reshape(-1,10),target.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    if step%2000==0:
      ra=evaluate(model,50000,0); ea=evaluate(model,50000,1)
      print(step,loss.item(),ra,ea,flush=True)
      if ra[0]==1 and ea[0]==1: torch.save(model.state_dict(),'/workspace/factor.pt')
