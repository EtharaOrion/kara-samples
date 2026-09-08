import sys, itertools, torch
sys.path.insert(0,'/workspace')
from submission import AdditionTransformer
from train import POW8,POW9
m=AdditionTransformer().cuda(); m.load_state_dict(torch.load('/workspace/refined_best.pt',weights_only=True)); m.eval()
vals=set()
for base in [10_000_000,11_111_111,20_000_000,40_000_000,49_999_999,50_000_000,50_000_001,88_888_888,89_999_999,90_000_000,98_888_888,99_000_000,99_900_000,99_990_000,99_999_000,99_999_900,99_999_990,99_999_999]:
 for d in range(-10,11):
  if 10_000_000<=base+d<=99_999_999: vals.add(base+d)
for lead in range(1,10):
 for run in range(1,8):
  p=10**run
  for suffix in [0,1,p//2,p-2,p-1]:
   vals.add(lead*10_000_000+suffix)
for d in range(1,10): vals.add(d*11_111_111)
vals=sorted(vals)
pairs=list(itertools.product(vals,vals))
# Include exhaustive leading combinations around complements and carry suffix lengths.
for la in range(1,10):
 for lb in range(1,10):
  for run in range(1,8):
   p=10**run
   for da,db in [(p-1,1),(p-9,9),(0,p-1),(p-1,0)]:
    pairs.append((min(99_999_999,la*10_000_000+da),min(99_999_999,lb*10_000_000+db)))
print('values',len(vals),'pairs',len(pairs))
good=0; failures=[]; minmargin=100.
for off in range(0,len(pairs),8192):
 chunk=pairs[off:off+8192]; a=torch.tensor([x for x,y in chunk],device='cuda'); b=torch.tensor([y for x,y in chunk],device='cuda')
 ad=(a[:,None]//POW8)%10;bd=(b[:,None]//POW8)%10;truth=(a+b)[:,None]//POW9%10
 seq=torch.empty(len(chunk),17,dtype=torch.long,device='cuda');seq[:,:16:2]=ad;seq[:,1:16:2]=bd;seq[:,16]=10; preds=[]
 with torch.no_grad():
  for _ in range(9):
   logits=m(seq)[:,-1]; top=logits.topk(2,-1).values; minmargin=min(minmargin,float((top[:,0]-top[:,1]).min())); d=logits.argmax(-1);preds.append(d);seq=torch.cat((seq,d[:,None]),1)
 pred=torch.stack(preds,1); ok=(pred==truth).all(1);good+=int(ok.sum())
 for i in (~ok).nonzero().flatten()[:max(0,20-len(failures))]: failures.append((chunk[int(i)],pred[int(i)].tolist(),truth[int(i)].tolist()))
print('EDGE',good,len(pairs),'margin',minmargin,'failures',failures)
