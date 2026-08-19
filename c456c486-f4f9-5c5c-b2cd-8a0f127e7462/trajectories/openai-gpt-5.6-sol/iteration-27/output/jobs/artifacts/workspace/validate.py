import sys, time, torch
sys.path.extend(['/usr/local/lib/python3.11/dist-packages','/usr/local/lib/python3.11/site-packages'])
import submission, train
m,meta=submission.build_model(); m=m.cuda().eval()
print('meta',meta,'params',sum(p.numel() for p in m.parameters()))
t=time.time()
for seed in (8101,8102,8103,8104):
 torch.manual_seed(seed)
 print('uniform',seed,train.random_errors(m,524288),flush=True)
for seed in (9101,9102,9103,9104):
 torch.manual_seed(seed)
 print('structured',seed,train.structured_errors(m,262144),flush=True)
print('systematic',train.systematic_errors(m),flush=True)
# Broader deterministic adversarial collection.
pairs=set(train.systematic_pairs()); B=train.BASE
for k in range(14):
 p=10**k
 for run in range(1,15-k):
  r=(10**run-1)*p
  for x in range(1,10):
   for pair in ((r,x*p),(x*p,r),(r,(10-x)*p),(B-1-r,r),(5*p,5*p),(5, p+5)):
    if 0<=pair[0]<B and 0<=pair[1]<B:pairs.add(pair)
for x in range(10):
 d=int(str(x)*14)
 pairs.add((d,d)); pairs.add((d,B-1-d)); pairs.add((d,1))
pairs=list(pairs); a=torch.tensor([x for x,y in pairs],device='cuda'); b=torch.tensor([y for x,y in pairs],device='cuda')
wrong=(train.decode(m,train.to_digits(a),train.to_digits(b))!=train.to_digits(a+b,15)).any(1)
print('broad_edge',int(wrong.sum()),len(pairs),[pairs[i] for i in wrong.nonzero().flatten().tolist()[:10]],flush=True)
# Input-dependent QK scores from identical positions under two operand pairs.
a1=train.to_digits(torch.tensor([12345678901234,99999999999999],device='cuda'))
b1=train.to_digits(torch.tensor([87654321098765,1],device='cuda'))
pad=torch.full((2,1),10,device='cuda',dtype=torch.long)
x=m.a_embed(torch.cat((a1,pad),1))+m.b_embed(torch.cat((b1,pad),1))+m.position[:15]
z=m.blocks[0].ln1(x); q,k,_=m.blocks[0].qkv(z).chunk(3,-1); q=q.view(2,15,2,5).transpose(1,2); k=k.view(2,15,2,5).transpose(1,2)
s=torch.matmul(q,k.transpose(-2,-1)); print('qk_input_delta',float((s[0]-s[1]).abs().max()),flush=True)
print('minutes',(time.time()-t)/60)
