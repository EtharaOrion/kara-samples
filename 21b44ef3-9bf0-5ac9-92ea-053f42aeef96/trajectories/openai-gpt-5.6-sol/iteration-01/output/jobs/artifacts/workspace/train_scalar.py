import math
import sys
sys.path = [p for p in sys.path if "openhands-venv" not in p]
sys.path.extend(["/usr/local/lib/python3.11/dist-packages", "/usr/lib/python3/dist-packages"])
import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.embedding = nn.Embedding(12, width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.ff1 = nn.Linear(width, hidden)
        self.ff2 = nn.Linear(hidden, width)
        self.output = nn.Linear(width, 1)
    def forward(self, tokens):
        x = self.embedding(tokens)
        q,k,v = self.qkv(x).chunk(3,-1)
        attn = F.softmax(q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1]), -1)
        x = x + self.projection(attn @ v)
        x = x + self.ff2(F.silu(self.ff1(x)))
        return self.output(x[:,2]).squeeze(-1)

def run(width, hidden, seed, steps):
    torch.manual_seed(seed); d='cuda'
    x=torch.tensor([(a,b,10+c) for c in range(2) for a in range(10) for b in range(10)],device=d)
    y=torch.tensor([a+b+c for c in range(2) for a in range(10) for b in range(10)],device=d,dtype=torch.float)
    m=Model(width,hidden).to(d); o=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-5)
    best=0
    for s in range(steps):
        o.zero_grad(set_to_none=True); pred=m(x); loss=F.mse_loss(pred,y); loss.backward(); o.step()
        with torch.no_grad(): acc=(m(x).round().clamp(0,19)==y).float().mean().item()
        if acc>best: best=acc
        if s%1000==0 or s==steps-1: print(s,loss.item(),acc,best,flush=True)
        if acc==1 and s>=2000:
            # optimize margin/precision for another 3000 steps
            if not hasattr(run,'perfect'): run.perfect=s
            if s-run.perfect>=3000: break
    m.eval(); torch.save(m.cpu().state_dict(),f'/workspace/scalar_{width}_{hidden}_{seed}.pt')
    with torch.no_grad():
        p=m.cpu()(x.cpu()); err=(p-y.cpu()).abs(); print('params',sum(v.numel() for v in m.parameters()),'acc',(err<.5).float().mean().item(),'maxerr',err.max().item(),'perfect_start',getattr(run,'perfect',None))
if __name__=='__main__': run(*map(int,sys.argv[1:]))
