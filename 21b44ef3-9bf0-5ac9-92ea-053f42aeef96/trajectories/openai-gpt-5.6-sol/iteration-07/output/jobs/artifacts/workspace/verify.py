import copy, sys, time
import torch
sys.path.insert(0, '/workspace')
import submission
from train import validate, predict, sample_batch, SPECIAL

model, metadata = submission.build_model()
print('parameters', sum(p.numel() for p in model.parameters()), metadata)
print('registered_float_params', all(any(p is q for p in model.parameters()) for q in model.parameters()))
print('cpu_examples', [(a,b,submission.add(model,a,b)) for a,b in [(10000000,10000000),(99999999,99999999),(50000000,50000000),(99999990,10000010),(12345678,87654321)]])
model.cuda()
torch.manual_seed(987654321)
t=time.time(); print('random_200k', validate(model, 200000, False), 'seconds', time.time()-t)
t=time.time(); print('structured_200k', validate(model, 200000, True), 'seconds', time.time()-t)
# Exhaustive Cartesian product over curated difficult values.
s = SPECIAL.cuda(); a=s.repeat_interleave(len(s)); b=s.repeat(len(s)); p,y=predict(model,a,b)
print('special_cartesian', int((p==y).all(1).sum()), '/', len(a))
# Dependence tests on the same 256 random complete predictions.
a=torch.randint(10000000,100000000,(256,),device='cuda'); b=torch.randint(10000000,100000000,(256,),device='cuda')
base,_=predict(model,a,b)
ablated=copy.deepcopy(model)
with torch.no_grad():
    for p0 in ablated.attn.parameters(): p0.zero_()
changed_attn=(predict(ablated,a,b)[0]!=base).any(1).sum().item()
corrupt=copy.deepcopy(model)
with torch.no_grad(): corrupt.output.weight.zero_()
changed_output=(predict(corrupt,a,b)[0]!=base).any(1).sum().item()
print('attention_ablation_changed', changed_attn, '/ 256')
print('output_corruption_changed', changed_output, '/ 256')
# Explicitly compare computed attention distributions for differing inputs.
with torch.no_grad():
    x1,_=sample_batch(1,torch.device('cuda'),0); x2,_=sample_batch(1,torch.device('cuda'),0)
    def weights(x):
        L=x.shape[1]; h=model.token(x)+model.position(torch.arange(L,device='cuda'))+model.passes.weight[0]
        z=model.norm1(h); mask=torch.triu(torch.ones(L,L,device='cuda',dtype=torch.bool),1)
        return model.attn(z,z,z,attn_mask=mask,need_weights=True,average_attn_weights=False)[1]
    print('attention_input_delta', (weights(x1)-weights(x2)).abs().max().item())
