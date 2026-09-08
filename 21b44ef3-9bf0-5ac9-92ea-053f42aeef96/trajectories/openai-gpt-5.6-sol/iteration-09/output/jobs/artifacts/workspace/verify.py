import importlib.util, random, torch
spec=importlib.util.spec_from_file_location('submission','/workspace/submission.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
m,meta=s.build_model(); print('params',sum(p.numel() for p in m.parameters()),meta)
r=random.Random(9182)
pairs=[(r.randint(10_000_000,99_999_999),r.randint(10_000_000,99_999_999)) for _ in range(10000)]
edges=[]
vals=[10_000_000,10_000_001,10_000_009,10_000_090,11_111_111,49_999_999,50_000_000,88_888_888,90_000_000,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999]
for a in vals:
 for b in vals: edges.append((a,b))
edges += [(a,100_000_000-a) for a in range(10_000_000,90_000_001,100003)]
def score(ps):
 bad=[]
 for a,b in ps:
  y=s.add(m,a,b)
  if y!=a+b: bad.append((a,b,y,a+b))
 return len(ps)-len(bad),len(ps),bad[:5]
print('random',score(pairs)); print('edges',score(edges))
base=[s.add(m,*x) for x in pairs[:100]]
state={k:v.clone() for k,v in m.state_dict().items()}
with torch.no_grad():
 m.block.attn.in_proj_weight.zero_(); m.block.attn.in_proj_bias.zero_(); m.block.attn.out_proj.weight.zero_(); m.block.attn.out_proj.bias.zero_()
abl=[s.add(m,*x) for x in pairs[:100]]; print('attention_ablation_changed',sum(a!=b for a,b in zip(base,abl)))
m.load_state_dict(state)
with torch.no_grad(): m.output.weight.zero_()
out=[s.add(m,*x) for x in pairs[:100]]; print('output_corruption_changed',sum(a!=b for a,b in zip(base,out)))
m.load_state_dict(state)
# Capture per-head attention for two operand sequences.
def toks(a,b):
 ad=[int(c) for c in str(a)[::-1]]; bd=[int(c) for c in str(b)[::-1]]
 return torch.tensor([[v for q in zip(ad,bd) for v in q]+[10]])
def weights(t):
 n=t.shape[1]; x=m.token(t)+m.position_left[:n]@m.position_right; z=m.block.norm1(x+m.passes.weight[0]); mask=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
 return m.block.attn(z,z,z,attn_mask=mask,need_weights=True,average_attn_weights=False)[1]
w1=weights(toks(*pairs[0])); w2=weights(toks(*pairs[1])); print('attention_input_delta',float((w1-w2).abs().max()))
