import sys, torch
sys.path.insert(0, '/workspace')
import submission
from train import exact, uniform, structured, digits

m, meta = submission.build_model()
print('parameters', sum(p.numel() for p in m.parameters()), meta)
m.cuda().eval()
print('uniform_524288_errors', exact(m, 524288, 'uniform'))
print('structured_262144_errors', exact(m, 262144, 'structured'))

# Deterministic long-carry, no-carry contrast, sparse 5+5/9+9, extremes.
cases = {(0,0),(10**14-1,10**14-1),(10**14-1,1),(10**14-1,0)}
for start in range(14):
    p = 10**start
    for length in range(1, 15-start):
        run = 10**(start+length)-p
        cases.add((run,p)); cases.add((p,run)); cases.add((run,max(0,p-1)))
    for d in (1,5,9): cases.add((d*p,d*p))
wrong=[]
for a,b in cases:
    got=submission.add(m,a,b)
    if got != a+b: wrong.append((a,b,got,a+b))
print('systematic',len(cases),'errors',len(wrong),wrong[:10])

# Explicit QK score dependence in block 1.
ad1=torch.zeros((1,14),dtype=torch.long,device='cuda'); bd1=torch.zeros_like(ad1)
ad2=torch.full((1,14),9,dtype=torch.long,device='cuda'); bd2=torch.arange(14,device='cuda').remainder(10).view(1,-1)
def scores(ad,bd):
    x=m.a_embed(ad)+m.b_embed(bd)
    start=(m.a_embed.weight[10]+m.b_embed.weight[10]).view(1,1,9)
    x=torch.cat((x,start),1)+m.position[:15].unsqueeze(0)
    z=m.block1.norm(x); q,k,_=m.block1.qkv(z).chunk(3,-1)
    q=q.view(1,15,3,3).transpose(1,2); k=k.view(1,15,3,3).transpose(1,2)
    return q@k.transpose(-2,-1)
print('qk_delta',float((scores(ad1,bd1)-scores(ad2,bd2)).abs().max()))
