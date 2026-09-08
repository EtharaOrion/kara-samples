import copy
import random
import sys
import time
import torch
sys.path.insert(0, '/workspace')
import submission

torch.manual_seed(12345)
random.seed(12345)
model, metadata = submission.build_model()
assert not model.training
assert sum(p.numel() for p in model.parameters()) == 6160
started = time.time()
cases = [(random.randint(10_000_000,99_999_999), random.randint(10_000_000,99_999_999)) for _ in range(3000)]
edge_values = [10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,50_000_000,88_888_888,89_999_999,90_000_000,98_888_888,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999]
edges = [(a,b) for a in edge_values for b in edge_values]
random_ok = sum(submission.add(model,a,b)==a+b for a,b in cases)
edge_ok = sum(submission.add(model,a,b)==a+b for a,b in edges)
print('accuracy', random_ok, '/', len(cases), 'edge', edge_ok, '/', len(edges), 'seconds', time.time()-started)

sample = cases[:100]
baseline = [submission.add(model,a,b) for a,b in sample]
ablated = copy.deepcopy(model)
with torch.no_grad():
    for block in ablated.blocks:
        block.attn.in_proj_weight.zero_()
        block.attn.in_proj_bias.zero_()
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
attention_changed = sum(x != submission.add(ablated,a,b) for x,(a,b) in zip(baseline,sample))
corrupt = copy.deepcopy(model)
with torch.no_grad(): corrupt.output.weight.copy_(corrupt.output.weight.roll(1,0))
output_changed = sum(x != submission.add(corrupt,a,b) for x,(a,b) in zip(baseline,sample))

# Directly confirm attention distributions vary with operand tokens.
def tokens(a,b):
    out=[]
    for x,y in zip(str(a)[::-1],str(b)[::-1]): out += [ord(x)-48,ord(y)-48]
    return torch.tensor([out+[10]])
s1=tokens(12345678,87654321); s2=tokens(98765432,11111111)
with torch.no_grad():
    x1=model.token(s1)+model.position.weight[:17]
    x2=model.token(s2)+model.position.weight[:17]
    n=model.blocks[0].norm1
    _,w1=model.blocks[0].attn(n(x1),n(x1),n(x1),need_weights=True,average_attn_weights=False)
    _,w2=model.blocks[0].attn(n(x2),n(x2),n(x2),need_weights=True,average_attn_weights=False)
print('attention_ablation_changed',attention_changed,'/100','output_corruption_changed',output_changed,'/100','attention_input_delta',float((w1-w2).abs().max()))
assert random_ok >= int(.99*len(cases)) and edge_ok >= int(.99*len(edges))
assert attention_changed > 0 and output_changed > 0 and (w1-w2).abs().max() > 1e-4
