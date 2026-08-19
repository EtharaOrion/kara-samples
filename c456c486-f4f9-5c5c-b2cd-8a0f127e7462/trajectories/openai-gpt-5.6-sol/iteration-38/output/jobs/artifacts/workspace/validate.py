import random,sys,torch
sys.path.insert(0,'/workspace');import submission
m,_=submission.build_model();m.load_state_dict(torch.load('/workspace/model.pt',weights_only=True));m.eval()
# CPU batched autoregressive random validation.
def batch_add(a,b):
 p=torch.arange(14);ad=(a[:,None]//(10**p))%10;bd=(b[:,None]//(10**p))%10
 s=torch.full((len(a),1),10,dtype=torch.long);ad=torch.cat((ad,s),1);bd=torch.cat((bd,s),1)
 out=torch.empty((len(a),0),dtype=torch.long)
 for _ in range(15):out=torch.cat((out,m(ad,bd,out).argmax(1,keepdim=True)),1)
 return (out*(10**torch.arange(15))).sum(1)
g=torch.Generator().manual_seed(991);errs=0
for _ in range(32):
 a=torch.randint(0,10**14,(4096,),generator=g);b=torch.randint(0,10**14,(4096,),generator=g);errs+=(batch_add(a,b)!=a+b).sum().item()
print('random errors',errs,'/131072')
cases=[(0,0),(10**14-1,0),(10**14-1,1),(10**14-1,10**14-1),(99999999999990,9),(5,5)]
for p in range(14):
 cases += [(5*10**p,5*10**p),(9*10**p,9*10**p),((10**(p+1)-1),1),((10**(p+1)-2),1)]
fail=[]
for a,b in cases:
 got=submission.add(m,a,b)
 if got!=a+b:fail.append((a,b,got,a+b))
print('systematic',len(fail),'/',len(cases),fail[:10])
# Input-dependent QK scores in first block.
def qk(a,b):
 p=torch.arange(14);ad=torch.cat((((torch.tensor([a])[:,None]//(10**p))%10),torch.tensor([[10]])),1);bd=torch.cat((((torch.tensor([b])[:,None]//(10**p))%10),torch.tensor([[10]])),1)
 x=m.a_embed(ad)+m.b_embed(bd)+m.position[:15];z=m.blocks[0].norm(x);q,k,_=m.blocks[0].qkv(z).chunk(3,-1);return q.view(1,15,3,3).transpose(1,2)@k.view(1,15,3,3).transpose(1,2).transpose(-2,-1)
print('qk delta',float((qk(123,456)-qk(987654321,111111111)).abs().max()))
