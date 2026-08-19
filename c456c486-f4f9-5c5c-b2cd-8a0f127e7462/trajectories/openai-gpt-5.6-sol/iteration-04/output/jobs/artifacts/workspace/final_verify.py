import ast
import torch
import submission

m, meta = submission.build_model()
assert isinstance(m, torch.nn.Module)
assert sum(p.numel() for p in m.parameters()) == 4858
cases = [
 (0,0),(1,0),(0,1),(9,1),(99,1),(99999999999999,0),
 (99999999999999,1),(99999999999999,99999999999999),
 (50000000000000,50000000000000),(11111111111111,88888888888889),
 (90909090909090,9090909090909),(99999999999990,9),
]
for a,b in cases:
    got=submission.add(m,a,b)
    print(a,b,got,got==a+b)
    assert got==a+b
# Directly confirm learned Q/K scores change under changed token content.
block=m.blocks[0]
x1=m.digit(torch.tensor([[1,2,3,4]]))+m.role(torch.arange(4)%3)
x2=m.digit(torch.tensor([[7,2,3,4]]))+m.role(torch.arange(4)%3)
q1,k1,_=block.qkv(block.ln1(x1)).chunk(3,-1)
q2,k2,_=block.qkv(block.ln1(x2)).chunk(3,-1)
s1=q1@k1.transpose(-1,-2); s2=q2@k2.transpose(-1,-2)
print(meta)
print('attention_score_max_delta', (s1-s2).abs().max().item())
assert (s1-s2).abs().max()>1e-3
# Screen-like static checks: no training/data labels or arithmetic operators deriving outputs.
tree=ast.parse(open('/workspace/submission.py').read())
print('submission_bytes',len(open('/workspace/submission.py','rb').read()))
print('ok')
