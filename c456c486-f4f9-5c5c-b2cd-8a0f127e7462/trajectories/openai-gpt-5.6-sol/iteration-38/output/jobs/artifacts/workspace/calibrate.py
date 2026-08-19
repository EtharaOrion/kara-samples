import sys,time,torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace');from submission import AdditionTransformer
D='cuda'; B=4096
m=AdditionTransformer().to(D);m.load_state_dict(torch.load('/workspace/model.pt',weights_only=True));m.train()
def digs(n,c):return (n[:,None]//(10**torch.arange(c,device=D)))%10
def batch():
 a=torch.randint(0,10**14,(B,),device=D);b=torch.randint(0,10**14,(B,),device=D)
 n=B//2; L=torch.randint(1,15,(n,),device=D); shift=(torch.rand(n,device=D)*(15-L)).long()
 base=10**L-1
 fam=torch.randint(0,4,(n,),device=D)
 # Exact carry chains and their non-carry contrasts, placed at varied columns.
 aa=torch.where(fam<2,base,base-1)*(10**shift)
 bb=torch.where(fam%2==0,torch.ones_like(base),torch.randint(1,10,(n,),device=D))*(10**shift)
 # family 3 gives isolated 9+9 boundary columns.
 iso=(fam==3);aa=torch.where(iso,9*(10**shift),aa);bb=torch.where(iso,9*(10**shift),bb)
 a[:n]=aa;b[:n]=bb
 y=digs(a+b,15);s=torch.full((B,1),10,device=D,dtype=torch.long)
 return torch.cat((digs(a,14),s),1),torch.cat((digs(b,14),s),1),y

def logits(ad,bd,y):
 x=torch.cat((m.a_embed(ad)+m.b_embed(bd),m.out_embed(y[:,:-1])),1)+m.position
 for z in m.blocks:x=z(x,m.mask)
 return m.head(m.final_norm(x[:,14:]))
o=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.001)
for step in range(1,16001):
 ad,bd,y=batch();z=logits(ad,bd,y);loss=F.cross_entropy(z.reshape(-1,10),y.reshape(-1));o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
 if step%2000==0:print(step,float(loss),(z.argmax(-1)!=y).any(1).sum().item(),flush=True)
torch.save(m.state_dict(),'/workspace/model.pt')
