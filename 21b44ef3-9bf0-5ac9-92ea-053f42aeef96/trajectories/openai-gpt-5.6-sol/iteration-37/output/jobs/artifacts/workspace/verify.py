import random,sys,torch
sys.path.insert(0,'/workspace'); import submission
m,_=submission.build_model(); m.cuda()

def batch_acc(n, structured=False):
 good=total=0
 for _ in range((n+9999)//10000):
  z=min(10000,n-total)
  if structured:
   k=torch.randint(1,8,(z,),device='cuda'); p=10**k
   a=torch.randint(10_000_000,100_000_000,(z,),device='cuda')
   b=(torch.randint(1,10,(z,),device='cuda')*10_000_000+(p-a.remainder(p)+torch.randint(-5,6,(z,),device='cuda')).remainder(p)).clamp(10_000_000,99_999_999)
  else:
   a=torch.randint(10_000_000,100_000_000,(z,),device='cuda'); b=torch.randint(10_000_000,100_000_000,(z,),device='cuda')
  aa=a.clone();bb=b.clone(); cols=[]
  for j in range(8): cols += [aa.remainder(10),bb.remainder(10)]; aa//=10;bb//=10
  seq=torch.stack(cols,1); seq=torch.cat((seq,torch.full((z,1),10,device='cuda')),1).long()
  target=a+b; value=torch.zeros_like(target)
  mult=1
  for j in range(9):
   d=m(seq)[:,-1].argmax(1); value += d*mult; mult*=10
   if j<8:seq=torch.cat((seq,d[:,None]),1)
  good+=int((value==target).sum());total+=z
 return good,total
print('params',sum(p.numel() for p in m.parameters()))
print('uniform',batch_acc(500000));print('carry',batch_acc(500000,True))
m.cpu(); random.seed(11); cases=[(random.randrange(10_000_000,100_000_000),random.randrange(10_000_000,100_000_000)) for _ in range(1000)]
patterns=[10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,40_000_009,50_000_000,89_999_999,90_000_000,99_000_001,99_899_999,99_999_900,99_999_999]
cases += [(a,b) for a in patterns for b in patterns]
good=sum(submission.add(m,a,b)==a+b for a,b in cases);print('api',good,len(cases))
base=[submission.add(m,a,b) for a,b in cases[:40]]
with torch.no_grad():
 for o in m.attn_out:o.weight.zero_()
abl=[submission.add(m,a,b) for a,b in cases[:40]]
print('attention_changed',sum(x!=y for x,y in zip(base,abl)))
m,_=submission.build_model()
with torch.no_grad():m.classifier.weight.zero_();m.classifier.bias.zero_()
out=[submission.add(m,a,b) for a,b in cases[:40]]
print('output_changed',sum(x!=y for x,y in zip(base,out)))
