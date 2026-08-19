import ast, random, sys, torch
sys.path.insert(0,'/workspace')
import submission
m,_=submission.build_model()
random.seed(161803); good=0
for i in range(300):
 a=random.randrange(10**14); b=random.randrange(10**14); good += submission.add(m,a,b)==a+b
print('exact',good/300)
# Confirm attention scores depend on token content.
b=m.blocks[0]
x1=m.digit(torch.tensor([[1,2,3,4]]))+m.role(torch.arange(4).remainder(3))
x2=m.digit(torch.tensor([[7,8,9,0]]))+m.role(torch.arange(4).remainder(3))
with torch.no_grad():
 q1,k1,_=b.qkv(b.n1(x1)).chunk(3,-1);q2,k2,_=b.qkv(b.n1(x2)).chunk(3,-1)
 s1=q1.view(1,4,2,8).transpose(1,2)@k1.view(1,4,2,8).transpose(1,2).transpose(-2,-1)
 s2=q2.view(1,4,2,8).transpose(1,2)@k2.view(1,4,2,8).transpose(1,2).transpose(-2,-1)
 print('qk_change',float((s1-s2).abs().max()))
print('params',sum(p.numel() for p in m.parameters()))
print('imports',[ast.unparse(n) for n in ast.parse(open('/workspace/submission.py').read()).body if isinstance(n,(ast.Import,ast.ImportFrom))])
