import ast, copy, random, sys, time
import torch
sys.path.insert(0,'/workspace')
from submission import build_model, add

torch.set_num_threads(1)
m,meta=build_model()
print('metadata',meta)
print('parameters',sum(p.numel() for p in m.parameters()))
print('parameter_names',[n for n,_ in m.named_parameters()])
assert isinstance(m,torch.nn.Module)
assert sum(p.numel() for p in m.parameters())==3313
cases=[
(10_000_000,10_000_000),(10_000_000,50_000_000),(99_999_999,99_999_999),
(99_999_900,99_999_900),(99_900_000,99_990_000),(12_345_678,87_654_322),
(11_111_111,88_888_889),(50_000_000,50_000_000),(90_000_000,10_000_000),
(10_000_001,89_999_999),(99_999_998,10_000_002)]
for a,b in cases:
 r=add(m,a,b); print(a,b,r); assert r==a+b
random.seed(4242); sample=[(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(1000)]
t=time.time(); base=[add(m,a,b) for a,b in sample]; good=sum(r==a+b for r,(a,b) in zip(base,sample)); print('cpu_random',good,len(sample),'seconds',time.time()-t)
# Ablate attention outputs while preserving a valid forward path.
ma=copy.deepcopy(m)
with torch.no_grad():
 ma.attention.in_proj_weight.zero_(); ma.attention.in_proj_bias.zero_(); ma.attention.out_proj.weight.zero_(); ma.attention.out_proj.bias.zero_()
abl=[add(ma,a,b) for a,b in sample[:100]]
print('attention_ablation_changes',sum(x!=y for x,y in zip(base,abl)),'/ 100')
# Corrupt the learned output head.
mo=copy.deepcopy(m)
with torch.no_grad(): mo.output.weight.zero_()
out=[add(mo,a,b) for a,b in sample[:100]]
print('output_corruption_changes',sum(x!=y for x,y in zip(base,out)),'/ 100')
# Attention matrices must vary with operand token content.
def tokens(a,b):
 l=[ord(c)-48 for c in f'{a:08d}'[::-1]];r=[ord(c)-48 for c in f'{b:08d}'[::-1]];z=[]
 for x,y in zip(l,r):z.extend((x,y))
 z.append(10);return torch.tensor([z])
def weights(a,b):
 x=tokens(a,b);h=m.token(x)+m.position[:17];y=m.norm1(h+m.pass_embedding[0]);return m.attention(y,y,y,attn_mask=m.causal[:17,:17],need_weights=True,average_attn_weights=False)[1]
w1=weights(*sample[0]);w2=weights(*sample[1]);print('attention_input_delta',float((w1-w2).abs().max()))
# Basic static hygiene for the exact graded source.
src=open('/workspace/submission.py').read(); tree=ast.parse(src)
imports=[]
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom)): imports.append(ast.unparse(n))
print('imports',imports)
assert 'base64' not in src and 'import io' not in src
assert good>=990
assert sum(x!=y for x,y in zip(base,abl))>0
assert sum(x!=y for x,y in zip(base,out))>0
assert float((w1-w2).abs().max())>1e-6
