import ast, random, sys, torch
sys.path.insert(0,'/workspace')
from submission import build_model, add
m,meta=build_model()
print('params',sum(p.numel() for p in m.parameters()),'metadata',meta)
random.seed(8123)
pairs=[(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(3000)]
pairs += [(10_000_000,10_000_000),(99_999_999,99_999_999),(40_000_009,89_999_999),(99_000_001,99_899_999),(10_000_091,90_000_099)]
base=[add(m,a,b) for a,b in pairs]
wrong=[(a,b,p,a+b) for (a,b),p in zip(pairs,base) if p!=a+b]
print('api',len(pairs)-len(wrong),'/',len(pairs),'failures',wrong[:5])
# Attention-output ablation preserves forward execution but removes attended values.
ablated,_=build_model()
with torch.no_grad():
 for layer in ablated.outputs: layer.weight.zero_()
changed=sum(add(ablated,a,b)!=p for (a,b),p in zip(pairs[:40],base[:40]))
print('attention_ablation_changed',changed,'/40')
corrupt,_=build_model()
with torch.no_grad(): corrupt.classifier.weight.zero_()
changed2=sum(add(corrupt,a,b)!=p for (a,b),p in zip(pairs[:40],base[:40]))
print('output_corruption_changed',changed2,'/40')
# Explicitly check attention distributions differ between inputs.
def attention(model,a,b):
 toks=[]
 for x,y in zip(reversed(str(a)),reversed(str(b))):toks.extend((int(x),int(y)))
 toks.append(10);t=torch.tensor([toks]);x=model.token(t)+(model.pos_left@model.pos_right)[:17]
 z=model.attn_norm(x);q=model.queries[0](z).view(1,17,4,5).transpose(1,2);k=model.key(z).unsqueeze(1)
 s=(q@k.transpose(-2,-1))*(5**-.5);mask=torch.ones(17,17,dtype=torch.bool).tril();return s.masked_fill(~mask,-torch.inf).softmax(-1)
delta=(attention(m,12_345_678,87_654_321)-attention(m,91_234_567,18_765_432)).abs().max().item()
print('attention_input_delta',delta)
imports=[]
for n in ast.walk(ast.parse(open('/workspace/submission.py').read())):
 if isinstance(n,(ast.Import,ast.ImportFrom)): imports.append(ast.unparse(n))
print('imports',imports)
