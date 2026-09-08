import copy, random, sys
import torch
sys.path.insert(0, '/workspace')
import submission

torch.manual_seed(2026); random.seed(2026); torch.set_num_threads(1)
model, metadata = submission.build_model()
count = sum(p.numel() for p in model.parameters())
print('metadata', metadata, 'count', count)
assert count == 2741
pairs = [(10_000_000,10_000_000),(99_999_999,99_999_999),(10_000_000,90_000_000),(50_000_000,50_000_000),(99_999_900,99_999_900),(12_345_678,87_654_321)]
pairs += [(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(1000)]
good = 0
base = []
for a,b in pairs:
    y = submission.add(model,a,b); base.append(y); good += y == a+b
print('public_add', good, '/', len(pairs))

ablated = copy.deepcopy(model)
with torch.no_grad():
    ablated.attn.in_proj_weight.zero_(); ablated.attn.in_proj_bias.zero_(); ablated.attn.out_proj.weight.zero_(); ablated.attn.out_proj.bias.zero_()
changed_attn = sum(submission.add(ablated,a,b) != y for (a,b),y in zip(pairs[:106],base[:106]))
corrupt = copy.deepcopy(model)
with torch.no_grad(): corrupt.output.weight.zero_()
changed_output = sum(submission.add(corrupt,a,b) != y for (a,b),y in zip(pairs[:106],base[:106]))
print('changed_attention',changed_attn,'/106','changed_output',changed_output,'/106')

# Capture per-head attention maps from the exact trained attention module.
def maps(a,b):
    left,right=str(a)[::-1],str(b)[::-1]
    tok=torch.tensor([[v for pair in zip(left,right) for v in (int(pair[0]),int(pair[1]))]+[10]])
    x=model.token(tok)+(model.pos_left[:17]@model.pos_right)
    x=x+(model.pass_left@model.pass_right)[0]
    y=model.norm1(x); mask=torch.triu(torch.ones(17,17,dtype=torch.bool),1)
    return model.attn(y,y,y,attn_mask=mask,need_weights=True,average_attn_weights=False)[1]
delta=(maps(12_345_678,87_654_321)-maps(98_765_432,11_111_111)).abs().max().item()
print('attention_input_delta',delta)
assert good/len(pairs) >= .99 and changed_attn and changed_output and delta > 1e-4
