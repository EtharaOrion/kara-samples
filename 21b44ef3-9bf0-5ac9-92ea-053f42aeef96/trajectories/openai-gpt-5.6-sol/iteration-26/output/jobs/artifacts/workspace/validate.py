import sys,random,torch
sys.path.insert(0,'/workspace')
import submission
m,meta=submission.build_model(); device='cuda';m.to(device)
print('params',sum(p.numel() for p in m.parameters()),meta,flush=True)

def toks(a,b):
 d=lambda x:[int(c) for c in str(x)[::-1]]
 aa,bb=d(a),d(b); z=[]
 for x,y in zip(aa,bb):z.extend((x,y))
 return z+[10]

def eval_pairs(pairs,chunk=4096):
 good=0;total=0;examples=[]
 with torch.no_grad():
  for q in range(0,len(pairs),chunk):
   part=pairs[q:q+chunk]; seq=torch.tensor([toks(a,b) for a,b in part],device=device)
   outs=[]
   for _ in range(9):
    d=m(seq)[:,-1].argmax(-1);outs.append(d);seq=torch.cat((seq,d[:,None]),1)
   pred=torch.stack(outs,1)
   powers=torch.tensor([10**i for i in range(9)],device=device)
   nums=(pred*powers).sum(1).cpu().tolist()
   for (a,b),v in zip(part,nums):
    if v==a+b:good+=1
    elif len(examples)<20:examples.append((a,b,v,a+b))
   total+=len(part)
 return good,total,examples
rng=random.Random(2604)
randoms=[(rng.randrange(10_000_000,100_000_000),rng.randrange(10_000_000,100_000_000)) for _ in range(500000)]
print('random AR',eval_pairs(randoms),flush=True)
vals=set([10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,11_111_111,20_000_000,40_000_009,49_999_999,50_000_000,50_000_001,89_999_999,90_000_000,98_999_999,99_000_000,99_000_001,99_899_999,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_998,99_999_999])
for k in range(1,8):
 p=10**k
 for lead in range(1,10):
  for off in [-2,-1,0,1,2]:
   v=lead*p+off
   if 10_000_000<=v<=99_999_999:vals.add(v)
edge=[(a,b) for a in vals for b in vals]
print('edge AR',len(vals),eval_pairs(edge),flush=True)
# direct API on representative subset
m.cpu(); api=randoms[:200]+edge[:200]
print('api',sum(submission.add(m,a,b)==a+b for a,b in api),len(api),flush=True)
# perturb attention outputs and classifier on same inputs
base=[submission.add(m,a,b) for a,b in api[:40]]
state=[x.weight.detach().clone() for x in m.outputs]
with torch.no_grad():
 for x in m.outputs:x.weight.zero_()
abl=[submission.add(m,a,b) for a,b in api[:40]]
with torch.no_grad():
 for x,w in zip(m.outputs,state):x.weight.copy_(w)
 head=m.head.weight.detach().clone()
with torch.no_grad():m.head.weight.zero_()
cor=[submission.add(m,a,b) for a,b in api[:40]]
with torch.no_grad():m.head.weight.copy_(head)
print('changed attention',sum(x!=y for x,y in zip(base,abl)),'head',sum(x!=y for x,y in zip(base,cor)),flush=True)
