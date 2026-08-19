import torch
from train import Adder, batch
m=Adder(False); m.load_state_dict(torch.load('/workspace/unshared.pt',weights_only=True)); m.eval()
N=100000
counts=torch.zeros(15,dtype=torch.long); errors=torch.zeros(15,dtype=torch.long)
carry_counts=torch.zeros(2,dtype=torch.long); carry_errors=torch.zeros(2,dtype=torch.long)
with torch.no_grad():
 for z in range(0,N,1000):
  t=batch(1000); seq=torch.empty((1000,0),dtype=torch.long); ps=[]
  carry=torch.zeros(1000,dtype=torch.long)
  for c in range(15):
   aa=t[:,3*c]; bb=t[:,3*c+1]; yy=t[:,3*c+2]
   seq=torch.cat((seq,t[:,3*c:3*c+2]),1); p=m(seq)[:,-1].argmax(-1); ps.append(p); seq=torch.cat((seq,p[:,None]),1)
   bad=p!=yy; errors[c]+=bad.sum(); counts[c]+=1000
   carry_counts.scatter_add_(0,carry,torch.ones_like(carry)); carry_errors.scatter_add_(0,carry,bad.long())
   carry=((aa+bb+carry)>=10).long()
 pred=torch.stack(ps,1); true=t[:,2::3]
print('col_error', (errors/counts).tolist())
print('carry_error', (carry_errors/carry_counts).tolist(), carry_counts.tolist())
