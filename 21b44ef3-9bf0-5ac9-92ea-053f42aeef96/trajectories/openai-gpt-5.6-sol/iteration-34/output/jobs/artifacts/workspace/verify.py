import sys,random,torch
sys.path.insert(0,'/workspace'); import submission
m,_=submission.build_model(); m=m.cuda()

def make(bs,structured=False):
 lo,hi=10_000_000,99_999_999
 a=torch.randint(lo,hi+1,(bs,),device='cuda',dtype=torch.long); b=torch.randint(lo,hi+1,(bs,),device='cuda',dtype=torch.long)
 if structured:
  L=torch.randint(1,8,(bs,),device='cuda'); p=(10**L).long(); typ=torch.randint(0,4,(bs,),device='cuda')
  a=torch.where(typ==0,torch.div(a,p,rounding_mode='floor')*p+p-1,a)
  b=torch.where(typ==0,torch.div(b,p,rounding_mode='floor')*p+1,b)
  a=torch.where(typ==1,torch.div(a,p,rounding_mode='floor')*p,a)
  b=torch.where(typ==1,torch.div(b,p,rounding_mode='floor')*p+p-1,b)
  a=a.clamp(lo,hi);b=b.clamp(lo,hi)
 def digs(z,n):
  z=z.clone()
  out=[]
  for _ in range(n):out.append(z%10);z//=10
  return torch.stack(out,1)
 da,db=digs(a,8),digs(b,8); y=digs(a+b,9)
 x=torch.empty(bs,17,device='cuda',dtype=torch.long);x[:,:16:2]=da;x[:,1:16:2]=db;x[:,16]=10
 return x,y

def ev(n,structured):
 good=0; margin=100.
 with torch.no_grad():
  for st in range(0,n,8192):
   x,y=make(min(8192,n-st),structured); ok=torch.ones(len(x),device='cuda',dtype=torch.bool)
   for j in range(9):
    z=m(x)[:,-1]; pred=z.argmax(1);ok&=pred.eq(y[:,j]); true=z.gather(1,y[:,j,None]).squeeze();z.scatter_(1,y[:,j,None],-1e9);margin=min(margin,(true-z.max(1).values).min().item())
    if j<8:x=torch.cat((x,pred[:,None]),1)
   good+=ok.sum().item()
 return good,n,margin
print('params',sum(p.numel() for p in m.parameters()))
print('uniform',ev(500000,False));print('structured',ev(500000,True))
# Broad deterministic CPU API suite.
m=m.cpu(); vals={10_000_000,10_000_001,10_000_009,10_000_010,10_000_099,10_000_100,10_000_999,10_001_000,10_009_999,10_010_000,10_099_999,10_100_000,10_999_999,11_111_111,20_000_000,40_000_009,49_999_999,50_000_000,50_000_001,88_888_888,89_999_999,90_000_000,98_999_999,99_000_001,99_899_999,99_900_000,99_999_900,99_999_990,99_999_999}
bad=[]
for a in vals:
 for b in vals:
  r=submission.add(m,a,b)
  if r!=a+b:bad.append((a,b,r,a+b))
print('edges',len(vals)**2-len(bad),len(vals)**2,'bad',bad[:5])
# Parameter corruption dependencies.
p=m.weights[9]; saved=p.detach().clone()
with torch.no_grad():p.zero_()
changed_att=sum(submission.add(m,a,b)!=a+b for a,b in [(random.randrange(10_000_000,100_000_000),random.randrange(10_000_000,100_000_000)) for _ in range(50)])
with torch.no_grad():p.copy_(saved);m.weights[14].zero_();m.weights[15].zero_()
changed_head=sum(submission.add(m,a,b)!=a+b for a,b in [(random.randrange(10_000_000,100_000_000),random.randrange(10_000_000,100_000_000)) for _ in range(50)])
print('ablations',changed_att,changed_head)
