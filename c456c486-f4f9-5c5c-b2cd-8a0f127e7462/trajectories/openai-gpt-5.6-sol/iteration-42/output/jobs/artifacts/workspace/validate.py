import sys
sys.path.insert(0, '/workspace')
import torch
from train import Model, digits, DEVICE

def decode(model,a,b):
    ad,bd=digits(a),digits(b); ad[:,14]=10; bd[:,14]=10
    out=torch.empty((len(a),0),dtype=torch.long,device=a.device)
    for _ in range(15):
        z=model(ad,bd,out)
        out=torch.cat((out,z[:,14+out.shape[1]].argmax(-1,keepdim=True)),1)
    return out

m=Model().cuda(); m.load_state_dict(torch.load('/workspace/model.pt',weights_only=True)); m.eval()
cases={(0,0),(99_999_999_999_999,99_999_999_999_999),(99_999_999_999_999,1),(50_000_000_000_000,50_000_000_000_000)}
for k in range(14):
    p=10**k
    for d in range(1,10):
        cases |= {(p*d,p*d),(p*5,p*5),(p*9,p*9),(p-1,1),(p*9-1,1),(p, p-1),(99_999_999_999_999-p+1,p-1)}
for start in range(14):
    for length in range(1,15-start):
        run=(10**(start+length)-10**start)
        cases |= {(run,10**start),(run-10**start,10**start),(run,10**start-1)}
cases=[x for x in cases if 0<=x[0]<100_000_000_000_000 and 0<=x[1]<100_000_000_000_000]
a=torch.tensor([x[0] for x in cases],device='cuda'); b=torch.tensor([x[1] for x in cases],device='cuda')
out=decode(m,a,b); target=digits(a+b)
bad=(out!=target).any(1)
print('systematic',int(bad.sum()),'/',len(cases))
if bad.any():
    for i in bad.nonzero()[:10]: print(cases[int(i)],out[int(i)].tolist(),target[int(i)].tolist())

# Q/K attention scores must change with digit content.
ad,bd=digits(torch.tensor([12345678901234,9876543210123],device='cuda')),digits(torch.tensor([11111111111111,22222222222222],device='cuda'))
ad[:,14]=10;bd[:,14]=10
x=m.a_emb(ad)+m.b_emb(bd); x=x+torch.nn.functional.pad(m.pos[:15],(0,7))[None]
z=m.attn1.norm(x); q,k,_=m.attn1.qkv(z).chunk(3,-1)
s=(q.view(2,15,3,3).transpose(1,2)@k.view(2,15,3,3).transpose(1,2).transpose(-2,-1))
print('qk_delta',float((s[0]-s[1]).abs().max()))

# Verify exported CPU model and direct add path on edge cases.
import submission
sm,md=submission.build_model()
print('parameters',sum(p.numel() for p in sm.parameters()),md)
for aa,bb in [(0,0),(1,2),(99_999_999_999_999,1),(99_999_999_999_999,99_999_999_999_999),(50_000_000_000_000,50_000_000_000_000),(12345678901234,87654321098765)]:
    got=submission.add(sm,aa,bb); print('cpu',aa,bb,got,got==aa+bb)
# Learned output parameters materially control predictions.
base=decode(m,a[:512],b[:512])
with torch.no_grad(): saved=m.head.weight.clone(); m.head.weight.zero_()
abl=decode(m,a[:512],b[:512])
print('head_ablation_changed',float((base!=abl).any(1).float().mean()))
