import sys, time
import torch
sys.path.insert(0, '/workspace')
from submission import AdditionTransformer
from train import LIMIT, carry_examples, pattern_examples, inputs, targets, to_digits

device=torch.device('cuda')
m=AdditionTransformer().to(device)
m.load_state_dict(torch.load('/workspace/model_final.pt',weights_only=True))
m.eval()

@torch.no_grad()
def check(name, total, family):
    errors=[]; done=0
    while done<total:
        n=min(16384,total-done)
        if family=='random':
            a=torch.randint(0,LIMIT,(n,),device=device); b=torch.randint(0,LIMIT,(n,),device=device)
        elif family=='carry': a,b=carry_examples(n,device)
        else: a,b=pattern_examples(n,device)
        pred=m(*inputs(a,b)).argmax(-1); bad=(pred!=targets(a,b)).any(1).nonzero().flatten()
        for i in bad[:max(0,20-len(errors))]: errors.append((a[i].item(),b[i].item()))
        done+=n
    print(name, 'errors',len(errors) if len(errors)<20 else '>=20','/',total,'samples',errors[:10],flush=True)

# Deterministic families across every decimal boundary and chain length.
pairs=set()
M=LIMIT-1
base=[0,1,2,5,9,10,11,99,100,101,999,1000,M,M-1,M//2,11111111111111,9999999999999]
for a in base:
 for b in base:
  if a<LIMIT and b<LIMIT: pairs.add((a,b))
for start in range(14):
 for length in range(1,15-start):
  p=10**start; run=(10**length-1)*p
  variants=[(run,p),(run,1),(M-run,p),(M-run,run),(5*run,5*p),(run,9*p)]
  for x,y in variants:
   if 0<=x<LIMIT and 0<=y<LIMIT: pairs.add((x,y)); pairs.add((y,x))
for k in range(1,15):
 p=10**k
 vals=[p-1,p,p+1,2*p-1,5*p-1,9*p-1]
 for x in vals:
  for y in [0,1,9,p-1,p]:
   if x<LIMIT and y<LIMIT: pairs.add((x,y)); pairs.add((y,x))
arr=list(pairs); a=torch.tensor([x for x,y in arr],device=device); b=torch.tensor([y for x,y in arr],device=device)
with torch.no_grad(): pred=m(*inputs(a,b)).argmax(-1); bad=(pred!=targets(a,b)).any(1)
print('systematic errors',bad.sum().item(),'/',len(arr),[arr[i] for i in bad.nonzero().flatten()[:10]],flush=True)

check('uniform',1048576,'random')
check('carry',1048576,'carry')
check('pattern',524288,'pattern')

# Input-dependent attention score check before the first softmax.
with torch.no_grad():
 a=torch.tensor([12345678901234,99999999999999],device=device); b=torch.tensor([43210987654321,1],device=device)
 l,r=inputs(a,b); x=m.digit(l)+m.digit(r)+m.position; z=m.norm1(x); q,k,_=m.qkv(z).chunk(3,-1)
 q=q.reshape(-1,15,2,6).transpose(1,2); k=k.reshape(-1,15,2,6).transpose(1,2)
 s=q@k.transpose(-2,-1)/(6**0.5)
 print('qk input max difference',(s[0]-s[1]).abs().max().item(),flush=True)

# CPU parity on deterministic set.
cpu=AdditionTransformer(); cpu.load_state_dict(torch.load('/workspace/model_final.pt',weights_only=True)); cpu.eval()
with torch.no_grad(): cpu_pred=cpu(*inputs(a.cpu(),b.cpu())).argmax(-1); gpu_pred=m(*inputs(a,b)).argmax(-1).cpu()
print('cpu_gpu_equal',torch.equal(cpu_pred,gpu_pred),flush=True)
