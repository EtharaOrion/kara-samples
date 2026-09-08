import copy, random, sys, torch
sys.path.insert(0,'/workspace')
import submission
m,meta=submission.build_model(); m=m.cuda()
POW=torch.tensor([10**i for i in range(9)],device='cuda',dtype=torch.long)
def check(count,structured=False,chunk=10000):
 good=0
 for z in range(0,count,chunk):
  n=min(chunk,count-z)
  a=torch.randint(10_000_000,100_000_000,(n,),device='cuda'); b=torch.randint(10_000_000,100_000_000,(n,),device='cuda')
  if structured:
   p=torch.randint(1,8,(n,),device='cuda'); base=POW[p]
   a=(a//base)*base+base-1; a=a.clamp(10_000_000,99_999_999)
  da=(a[:,None]//POW[None,:8])%10; db=(b[:,None]//POW[None,:8])%10
  toks=torch.cat((torch.stack((da,db),2).reshape(n,16),torch.full((n,1),10,device='cuda')),1)
  out=[]
  for _ in range(9):
   d=m(toks)[:,-1].argmax(1);out.append(d);toks=torch.cat((toks,d[:,None]),1)
  val=(torch.stack(out,1)*POW).sum(1)
  good+=int((val==a+b).sum())
 return good
print('meta',meta,'params',sum(p.numel() for p in m.parameters()))
print('random',check(500000),'structured',check(500000,True))
# direct edge grid CPU
m=m.cpu(); vals=[10_000_000,10_000_001,10_000_009,10_000_099,10_000_999,10_009_999,10_099_999,10_999_999,11_111_111,40_000_009,49_999_999,50_000_000,89_999_999,90_000_000,98_999_999,99_000_001,99_899_999,99_999_900,99_999_990,99_999_998,99_999_999]
bad=[]
for a in vals:
 for b in vals:
  r=submission.add(m,a,b)
  if r!=a+b:bad.append((a,b,r,a+b))
print('edge',len(vals)**2-len(bad),len(vals)**2,'bad',bad[:5])
# model dependence under output and attention corruption
random.seed(7); pairs=[(random.randrange(10_000_000,100_000_000),random.randrange(10_000_000,100_000_000)) for _ in range(30)]
base=[submission.add(m,*x) for x in pairs]
out=copy.deepcopy(m)
with torch.no_grad():out.head.weight.zero_();out.head.bias.zero_()
print('head_changed',sum(submission.add(out,*x)!=y for x,y in zip(pairs,base)))
abl=copy.deepcopy(m)
with torch.no_grad():
 for layer in abl.o:layer.weight.zero_()
print('attention_changed',sum(submission.add(abl,*x)!=y for x,y in zip(pairs,base)))
# Attention-score input dependence at layer zero.
def scores(a,b):
 da=[int(c) for c in str(a)[::-1]];db=[int(c) for c in str(b)[::-1]];t=[]
 for x,y in zip(da,db):t += [x,y]
 t=torch.tensor([t+[10]])
 with torch.no_grad():
  x=m.token(t)+m.position[:17];z=m.attn_norm[0](x);q=m.q[0](z).view(1,17,4,5).transpose(1,2);k=m.k[0](z).view(1,17,4,5).transpose(1,2)
  return (q@k.transpose(-2,-1))/5**.5
print('attention_score_delta',float((scores(*pairs[0])-scores(*pairs[1])).abs().max()))
