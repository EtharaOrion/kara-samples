import random, sys, torch
sys.path.insert(0,'/workspace')
import submission
m,_=submission.build_model(); m.cuda()

def batch(n):
 a=torch.randint(10_000_000,100_000_000,(n,),device='cuda'); b=torch.randint(10_000_000,100_000_000,(n,),device='cuda')
 p8=torch.tensor([1,10,100,1000,10000,100000,1000000,10000000],device='cuda'); ad=(a[:,None]//p8)%10; bd=(b[:,None]//p8)%10
 seq=torch.cat((torch.stack((ad,bd),2).reshape(n,16),torch.full((n,1),10,device='cuda')),1).long(); total=a+b; p9=torch.cat((p8,torch.tensor([100000000],device='cuda'))); target=(total[:,None]//p9)%10; pred=[]
 with torch.no_grad():
  for _ in range(9):
   d=m(seq)[:,-1].argmax(1); pred.append(d); seq=torch.cat((seq,d[:,None]),1)
 return int((torch.stack(pred,1)==target).all(1).sum()),n
print('params',sum(p.numel() for p in m.parameters()))
good=total=0
for _ in range(10):
 g,n=batch(10000); good+=g; total+=n
print('gpu_random',good,total)
m.cpu(); random.seed(77)
cases=[(random.randint(10_000_000,99_999_999),random.randint(10_000_000,99_999_999)) for _ in range(1000)]
edge=[10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,11_111_111,40_000_009,49_999_999,50_000_000,89_999_999,90_000_000,99_000_001,99_899_999,99_999_900,99_999_990,99_999_999]
cases += [(a,b) for a in edge for b in edge]
base=[submission.add(m,a,b) for a,b in cases]; print('cpu_api',sum(x==a+b for x,(a,b) in zip(base,cases)),len(cases))
# Ablate all attention outputs, then classifier.
with torch.no_grad():
 saved=[x.weight.clone() for x in m.o]
 for x in m.o: x.weight.zero_()
att=[submission.add(m,a,b) for a,b in cases[:40]]
with torch.no_grad():
 for x,w in zip(m.o,saved): x.weight.copy_(w)
 savedc=m.classifier.weight.clone(); m.classifier.weight.zero_()
cls=[submission.add(m,a,b) for a,b in cases[:40]]
print('attention_changed',sum(x!=y for x,y in zip(base[:40],att)),'classifier_changed',sum(x!=y for x,y in zip(base[:40],cls)))
