import os,sys,random
os.environ.pop('PYTHONPATH',None);sys.path.insert(0,'/workspace')
import torch,submission
m,_=submission.build_model()
@torch.inference_mode()
def beam(a,b,width=3):
 left=list(map(int,reversed(f'{a:015d}')));right=list(map(int,reversed(f'{b:015d}')))
 beams=[([],torch.tensor(0.0))]
 for place in range(15):
  expanded=[]
  for prior,score in beams:
   tok=[]
   for j,d in enumerate(prior):tok.extend((left[j],right[j],d))
   tok.extend((left[place],right[place]))
   lp=m(torch.tensor(tok).unsqueeze(0))[0,-1].log_softmax(-1)
   vals,inds=lp.topk(width)
   for val,ind in zip(vals,inds):expanded.append((prior+[int(ind)],torch.stack((score,val)).sum()))
  expanded.sort(key=lambda x:float(x[1]),reverse=True);beams=expanded[:width]
 return int(''.join(map(str,reversed(beams[0][0]))))
r=random.Random(78123)
for w in (2,3,5):
 ok=0
 for i in range(3000):
  a=r.randrange(10**14);b=r.randrange(10**14);ok+=beam(a,b,w)==a+b
 print(w,ok/3000)
