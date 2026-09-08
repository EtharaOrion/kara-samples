import sys,random,torch
sys.path.insert(0,'/workspace'); import submission
m,_=submission.build_model(); m=m

def make(bs,structured=False):
 lo,hi=10_000_000,99_999_999
 a=torch.randint(lo,hi+1,(bs,),device='cpu',dtype=torch.long); b=torch.randint(lo,hi+1,(bs,),device='cpu',dtype=torch.long)
 if structured:
  L=torch.randint(1,8,(bs,),device='cpu'); p=(10**L).long(); typ=torch.randint(0,4,(bs,),device='cpu')
  a=torch.where(typ==0,torch.div(a,p,rounding_mode='floor')*p+p-1,a)
  b=torch.where(typ==0,torch.div(b,p,rounding_mode='floor')*p+1,b)
  a=torch.where(typ==1,torch.div(a,p,rounding_mode='floor')*p,a)
  b=torch.where(typ==1,torch.div(b,p,rounding_mode='floor')*p+p-1,b)
  a=a.clamp(lo,hi);b=b.clamp(lo,hi)
 def digs(z,n):
  out=[]
  for _ in range(n):out.append(z%10);z//=10
  return torch.stack(out,1)
 da,db=digs(a,8),digs(b,8); y=digs(a+b,9)
 x=torch.empty(bs,17,device='cpu',dtype=torch.long);x[:,:16:2]=da;x[:,1:16:2]=db;x[:,16]=10
 return x,y

def ev(n,structured):
 good=0; margin=100.
 with torch.no_grad():
  for st in range(0,n,8192):
   x,y=make(min(8192,n-st),structured); ok=torch.ones(len(x),device='cpu',dtype=torch.bool)
   for j in range(9):
    z=m(x)[:,-1]; pred=z.argmax(1);ok&=pred.eq(y[:,j]); true=z.gather(1,y[:,j,None]).squeeze();z.scatter_(1,y[:,j,None],-1e9);margin=min(margin,(true-z.max(1).values).min().item())
    if j<8:x=torch.cat((x,pred[:,None]),1)
   good+=ok.sum().item()
 return good,n,margin
print('params',sum(p.numel() for p in m.parameters()))
print('uniform',ev(20000,False));print('structured',ev(20000,True))
