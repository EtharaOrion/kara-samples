import sys
sys.path.extend(['/usr/local/lib/python3.11/dist-packages','/workspace'])
import torch
import submission

def digs(x):
 p=torch.tensor([10**i for i in range(9)],device=x.device); return (x[:,None]//p%10).long()
def check(n, mode):
 good=total=0; m=model.cuda()
 with torch.no_grad():
  for _ in range((n+9999)//10000):
   z=min(10000,n-total)
   a=torch.randint(10_000_000,100_000_000,(z,),device='cuda'); b=torch.randint(10_000_000,100_000_000,(z,),device='cuda')
   if mode:
    k=torch.randint(1,9,(z,),device='cuda'); p=10**k; a=(a//p)*p+p-1; a=a.clamp(10_000_000,99_999_999)
   da,db=digs(a),digs(b); seq=torch.empty(z,17,dtype=torch.long,device='cuda');seq[:,0:16:2]=da[:,:8];seq[:,1:16:2]=db[:,:8];seq[:,16]=10; out=[]
   for j in range(9): d=m(seq)[:,-1].argmax(1);out.append(d);seq=torch.cat((seq,d[:,None]),1)
   good+=int((torch.stack(out,1)==digs(a+b)).all(1).sum());total+=z
 return good,total
model,meta=submission.build_model(); print('PARAMS',sum(p.numel() for p in model.parameters()),meta)
print('RANDOM',check(500000,0)); print('LONG9',check(500000,1))
model.cpu()
cases=[]
vals=[10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,11_111_111,49_999_999,50_000_000,89_999_999,90_000_000,98_999_999,99_000_001,99_999_990,99_999_999]
for a in vals:
 for b in vals: cases.append((a,b))
ok=sum(submission.add(model,a,b)==a+b for a,b in cases); print('EDGE',ok,len(cases))
base=[submission.add(model,a,b) for a,b in cases[:40]]
with torch.no_grad():
 saved=[p.clone() for p in model.attn.parameters()]
 for p in model.attn.parameters(): p.zero_()
abl=[submission.add(model,a,b) for a,b in cases[:40]]
with torch.no_grad():
 for p,s in zip(model.attn.parameters(),saved): p.copy_(s)
 model.head_weight.zero_();model.head_bias.zero_()
cor=[submission.add(model,a,b) for a,b in cases[:40]]
print('ABLATION_CHANGED',sum(x!=y for x,y in zip(base,abl)),'HEAD_CHANGED',sum(x!=y for x,y in zip(base,cor)))
