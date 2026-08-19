import sys, random, torch
sys.path.insert(0,'/workspace')
import submission
from train import evaluate
m,meta=submission.build_model(); m.cuda()
print(meta)
print('uniform_1m',evaluate(m,1048576,0))
print('mixed_1m',evaluate(m,1048576,.75))
# Comprehensive carry/noncarry/sparse/repeated edge suite via public add path on CPU.
m.cpu(); cases=set()
L=100_000_000_000_000
seeds=[0,1,2,5,9,10,11,99,L-1,L-2,L//2,11111111111111,9999999999999]
for x in seeds:
 for y in seeds:
  if x<L and y<L: cases.add((x,y))
for p in range(14):
 q=10**p
 for length in range(1,15-p):
  span=10**length
  for da,db in [(span-1,1),(span-2,1),(span-1,span-1),(5,5),(9,9),(4,5)]:
   a=da*q; b=db*q
   if a<L and b<L: cases.add((a,b)); cases.add((b,a))
for p in range(14):
 q=10**p
 for da in range(10):
  for db in range(10): cases.add((da*q,db*q))
errors=[]
for a,b in cases:
 got=submission.add(m,a,b)
 if got != a+b: errors.append((a,b,got,a+b))
print('edges',len(cases),'errors',len(errors),errors[:10])
# Compare QK scores for two tokenized inputs.
def scores(a):
 ad=torch.tensor([submission._digits(a)])
 bd=torch.tensor([submission._digits(0)])
 x=m.a_embed(ad)+m.b_embed(bd)
 x=x+torch.nn.functional.pad(m.position[:15],(0,5))
 y=m.block1.norm(x); q,k,_=m.block1.qkv(y).chunk(3,-1)
 return q.view(1,15,3,3)@k.view(1,15,3,3).transpose(-2,-1)
print('qk_delta',float((scores(123456789)-scores(987654321)).abs().max()))
