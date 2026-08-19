import copy, random, sys, torch
sys.path.insert(0, '/workspace')
import submission
m, meta = submission.build_model()
assert sum(p.numel() for p in m.parameters()) == 1533

cases = {(0,0),(99_999_999_999_999,0),(99_999_999_999_999,1),(99_999_999_999_999,99_999_999_999_999),(50_000_000_000_000,50_000_000_000_000)}
for k in range(14):
    p=10**k
    for x,y in [(p-1,1),(5*p,5*p),(9*p,9*p),(9*p,p),(8*p,p),(p,0),(p-1,0),(10*p-1,1)]:
        if 0 <= x < 10**14 and 0 <= y < 10**14: cases.add((x,y)); cases.add((y,x))
for length in range(1,15):
    n=10**length-1
    cases.add((n,1)); cases.add((1,n)); cases.add((n,0))
random.seed(9182)
for _ in range(1000):
    start=random.randrange(14); length=random.randrange(1,15-start)
    run=(10**length-1)*10**start
    low=random.randrange(10**start) if start else 0
    cases.add((run+low,10**start))
    cases.add((run+low,0))
errors=[]
for a,b in cases:
    got=submission.add(m,a,b)
    if got != a+b: errors.append((a,b,got,a+b))
print('params',sum(p.numel() for p in m.parameters()),'meta',meta,'systematic',len(cases),'errors',len(errors),errors[:10])
# Compute first-block QK scores directly on two differing complete source sequences.
a1=torch.tensor([[0]*14]); b1=torch.tensor([[0]*14])
a2=torch.tensor([[9,1,8,2,7,3,6,4,5,0,9,1,8,2]]); b2=torch.tensor([[1,8,2,7,3,6,4,5,0,9,1,8,2,7]])
def qks(a,b):
    stop=torch.full((1,1),10,dtype=torch.long)
    x=m.a_embed(torch.cat((a,stop),1))+m.b_embed(torch.cat((b,stop),1))+m.position[:15]
    z=m.blocks[0].norm1(x); q,k,_=m.blocks[0].qkv(z).chunk(3,-1)
    q=q.view(1,15,3,3).transpose(1,2); k=k.view(1,15,3,3).transpose(1,2)
    return q@k.transpose(-2,-1)
print('qk_max_change',float((qks(a1,b1)-qks(a2,b2)).abs().max()))
# Ensure output materially depends on learned parameters.
probe=(12_345_678_901_234,87_654_321_098_765)
before=submission.add(m,*probe)
with torch.no_grad(): m.head.weight.zero_(); m.head.bias.zero_()
after=submission.add(m,*probe)
print('perturbation',probe,before,after,'changed',before!=after)
