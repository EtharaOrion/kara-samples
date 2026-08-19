import importlib, random, sys, time
import torch
sys.path.insert(0, '/workspace')
import submission, train

torch.manual_seed(246810)
device=torch.device('cuda')
model,meta=submission.build_model(); model=model.to(device)
print('metadata',meta,'actual',sum(p.numel() for p in model.parameters()),'weights',len(submission._WEIGHTS))
t=time.time()
for kind,count in [('random',1048576),('mixed',524288),('broad',524288),('boundary',524288)]:
    e=train.evaluate(model,count,kind,device,chunk=32768)
    print(kind,e,'/',count,'elapsed',round(time.time()-t,1),flush=True)
e,n=train.systematic_errors(model,device); print('systematic',e,'/',n)
# Input-dependent QK score audit at source position 3.
a=torch.tensor([12345678901234,98765432109876],device=device)
b=torch.tensor([11111111111111,22222222222222],device=device)
ad,bd,target=train.model_inputs(a,b)
z=model.blocks[0].norm1(torch.add(torch.add(model.a_embedding(ad),model.b_embedding(bd)),model.position[:15]))
qkv=model.blocks[0].qkv(z).view(2,15,3,3,3)
q,k,_=qkv.unbind(2)
s=torch.einsum('blhd,bmhd->bhlm',q,k)/(3**0.5)
print('qk_max_difference',float((s[0]-s[1]).abs().max()))
# Perturbation audit: destroy output head and ensure outputs change.
p0=train.predict(model,a,b)
saved=model.head.weight.detach().clone(); model.head.weight.zero_(); p1=train.predict(model,a,b); model.head.weight.copy_(saved)
print('perturb_changed',bool((p0!=p1).any()),'predictions',p0.tolist())
# Public API on CPU, including edge cases and fresh random values.
model=model.cpu()
cases=[(0,0),(99999999999999,1),(99999999999999,99999999999999),(50000000000000,50000000000000),(9000000000000,9000000000000)]
r=random.Random(77)
cases += [(r.randrange(10**14),r.randrange(10**14)) for _ in range(100)]
fail=[]
for x,y in cases:
    got=submission.add(model,x,y)
    if got!=x+y: fail.append((x,y,got,x+y))
print('cpu_add',len(fail),'errors/',len(cases),fail[:3])
