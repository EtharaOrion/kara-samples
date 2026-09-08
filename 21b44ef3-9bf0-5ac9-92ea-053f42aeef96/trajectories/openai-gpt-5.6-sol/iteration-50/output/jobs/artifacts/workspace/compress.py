from itertools import combinations
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from train import AddTransformer, batch, accuracy

D=20
ck=torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)
base=AddTransformer(2); base.load_state_dict(ck['state']); base.eval()
s=base.state_dict()

# Fold affine feed-forward and output LayerNorms into their downstream maps.
f1w=s['f1.weight']*s['fn.weight']; f1b=s['f1.weight'].matmul(s['fn.bias'])
outw=s['out.weight']*s['outn.weight']; outb=s['out.weight'].matmul(s['outn.bias'])
# Reference-class softmax gauge, then exploit zero-sum non-affine LayerNorm outputs.
outw=outw[:9]-outw[9]; outb=outb[:9]-outb[9]
outw=outw-outw[:,-1,None]
f1w=f1w-f1w[:,-1,None]

# Remove the constant positional vector by absorbing it into every token embedding.
A=s['pa']-s['pa'][0]
tok=s['tok.weight']+s['pa'][0].matmul(s['pb'])
# Per-token all-ones translations are invisible to every subsequent LayerNorm.
tok=tok-tok[:,-1,None]
# Affine rank-2 gauge: row 0 is zero and two independent rows are the identity.
best=max(combinations(range(1,25),2),key=lambda ij:abs(torch.linalg.det(A[list(ij)]).item()))
M=A[list(best)]; ai=A.matmul(torch.linalg.inv(M)); pb=M.matmul(s['pb'])
# Position-dependent all-ones offsets are likewise removed by the first LayerNorm.
pb=pb-pb[:,-1,None]
pos_free=torch.stack([ai[i] for i in range(25) if i not in (0,*best)])

# Fold the affine attention LayerNorm into Q/K/V. The resulting input is a
# non-affine LayerNorm output with zero coordinate sum, so one column per map
# is represented as fixed zero. K's folded bias is softmax-invariant.
gamma=s['an.weight']; beta=s['an.bias']; drop=19
k0=s['k.weight']; v0=s['v.weight']
k=k0*gamma; v=v0*gamma
vb=v0.matmul(beta)
k=k-k[:,drop,None]
v=v-v[:,drop,None]

def basis_fix(w,excluded=()):
    choices=[c for c in range(D) if c not in excluded]
    cols=max(combinations(choices,5),key=lambda c:abs(torch.linalg.det(w[:,list(c)]).item()))
    B=w[:,list(cols)]
    return torch.linalg.solve(B,w),cols,B

kfix,kcols,KB=basis_fix(k,(drop,))
vfix,vcols,VB=basis_fix(v,(drop,))
# q'=q KB and value coordinates v'=v VB^-T, compensated in each output projection.
qw=[]; qb=[]; ow=[]
block=torch.block_diag(VB,VB,VB,VB)
for layer in range(2):
    q0=s[f'q.{layer}.weight']; q=q0*gamma
    qbias=q0.matmul(beta).reshape(4,5).matmul(KB)
    q=q-q[:,drop,None]
    qw.append(torch.einsum('ij,hjk->hik',KB.T,q.reshape(4,5,20)).reshape(20,20))
    qb.append(qbias.reshape(20))
    o=s[f'o.{layer}.weight'].matmul(block)
    # Residual-stream all-ones updates are invisible to all subsequent LayerNorms.
    o=o-o[-1,None]
    ow.append(o[:19])
f2=s['f2.weight']-s['f2.weight'][-1,None]
kfree=torch.stack([kfix[:,i] for i in range(D) if i not in (*kcols,drop)],1)
vfree=torch.stack([vfix[:,i] for i in range(D) if i not in (*vcols,drop)],1)
vbias=torch.linalg.solve(VB,vb)

DATA={
 'tok':tok[:,:19], 'pos':pos_free, 'pb':pb[:,:19], 'k':kfree, 'v':vfree, 'vb':vbias,
 'q0':qw[0][:,:19], 'qb0':qb[0], 'q1':qw[1][:,:19], 'qb1':qb[1], 'o0':ow[0], 'o1':ow[1],
 'f1w':f1w[:,:19], 'f1b':f1b, 'f2':f2[:19],
 'outw':outw[:,:19], 'outb':outb,
}
META={'pos_fixed':best,'k_fixed':kcols,'k_zero':drop,'v_fixed':vcols,'v_zero':drop}

class Compressed(nn.Module):
    def __init__(self,data):
        super().__init__()
        for n,t in data.items(): setattr(self,n,nn.Parameter(t.clone()))
    def expand(self):
        tok=torch.cat((self.tok,torch.zeros(11,1,device=self.tok.device)),1)
        rows=[]; fi=META['pos_fixed']
        free=iter(self.pos.unbind())
        for i in range(25):
            if i==0: rows.append(torch.zeros(2,device=tok.device))
            elif i==fi[0]: rows.append(torch.tensor([1.,0.],device=tok.device))
            elif i==fi[1]: rows.append(torch.tensor([0.,1.],device=tok.device))
            else: rows.append(next(free))
        pb=torch.cat((self.pb,torch.zeros(2,1,device=tok.device)),1)
        pos=torch.stack(rows).matmul(pb)
        def projection(free,fixed,zero=None):
            columns=[]; it=iter(free.unbind(1)); eye=torch.eye(5,device=tok.device)
            for i in range(20):
                if zero is not None and i==zero: columns.append(torch.zeros(5,device=tok.device))
                elif i in fixed: columns.append(eye[:,fixed.index(i)])
                else: columns.append(next(it))
            return torch.stack(columns,1)
        k=projection(self.k,list(META['k_fixed']),META['k_zero'])
        v=projection(self.v,list(META['v_fixed']),META['v_zero'])
        f1w=torch.cat((self.f1w,torch.zeros(2,1,device=tok.device)),1)
        outw=torch.cat((self.outw,torch.zeros(9,1,device=tok.device)),1)
        return tok,pos,k,v,f1w,outw
    def forward(self,t):
        tok,pos,kmat,vmat,f1w,outw=self.expand()
        x=F.embedding(t,tok)+pos[:t.shape[1]]
        f2=torch.cat((self.f2,torch.zeros(1,2,device=x.device)),0)
        for qfree,qb,ofree in ((self.q0,self.qb0,self.o0),(self.q1,self.qb1,self.o1)):
            o=torch.cat((ofree,torch.zeros(1,20,device=x.device)),0)
            q=torch.cat((qfree,torch.zeros(20,1,device=x.device)),1)
            z=F.layer_norm(x,(20,)); B,T,_=z.shape
            qq=F.linear(z,q,qb).view(B,T,4,5).transpose(1,2)
            kk=F.linear(z,kmat).view(B,1,T,5); vv=F.linear(z,vmat,self.vb).view(B,1,T,5)
            y=F.scaled_dot_product_attention(qq,kk,vv,is_causal=True)
            x=x+F.linear(y.transpose(1,2).reshape(B,T,20),o)
            n=F.layer_norm(x,(20,)); x=x+F.linear(F.gelu(F.linear(n,f1w,self.f1b)),f2)
        n=F.layer_norm(x,(20,)); nine=F.linear(n,outw,self.outb)
        return torch.cat((nine,torch.zeros((*nine.shape[:-1],1),device=nine.device)),2)

comp=Compressed(DATA).eval()
print('parameter count',sum(p.numel() for p in comp.parameters()),META)
with torch.no_grad():
    x,_=batch(10000,0.5,'cpu')
    a=base(x); c=comp(x)
    print('max logit diff (relative)',((a[...,:9]-a[...,9:])-c[...,:9]).abs().max().item(),
          'prediction differences',(a.argmax(-1)!=c.argmax(-1)).sum().item())

def export(path='/workspace/submission.py'):
    shapes={n:list(t.shape) for n,t in DATA.items()}; vals=[]
    for t in DATA.values(): vals.extend(t.half().reshape(-1).tolist())
    src=HEAD.replace('__META__',repr(META))+'\n_SHAPES='+repr(shapes)+'\n_VALUES='+repr(vals)+'\n'+TAIL
    Path(path).write_text(src)

HEAD='''import torch
from torch import nn
import torch.nn.functional as F
_META=__META__
class AdditionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        for name,shape in _SHAPES.items(): setattr(self,name,nn.Parameter(torch.empty(shape)))
    def _expand(self):
        z11=torch.zeros(11,1,device=self.tok.device); tok=torch.cat((self.tok,z11),1)
        rows=[]; free=iter(self.pos.unbind()); fixed=_META['pos_fixed']
        for i in range(25):
            if i==0: rows.append(torch.zeros(2,device=tok.device))
            elif i==fixed[0]: rows.append(torch.tensor([1.,0.],device=tok.device))
            elif i==fixed[1]: rows.append(torch.tensor([0.,1.],device=tok.device))
            else: rows.append(next(free))
        pb=torch.cat((self.pb,torch.zeros(2,1,device=tok.device)),1)
        pos=torch.stack(rows).matmul(pb); eye=torch.eye(5,device=tok.device)
        def projection(free,fixed,zero=-1):
            columns=[]; it=iter(free.unbind(1))
            for i in range(20):
                if i==zero: columns.append(torch.zeros(5,device=tok.device))
                elif i in fixed: columns.append(eye[:,fixed.index(i)])
                else: columns.append(next(it))
            return torch.stack(columns,1)
        k=projection(self.k,list(_META['k_fixed']),_META['k_zero'])
        v=projection(self.v,list(_META['v_fixed']),_META['v_zero'])
        f1=torch.cat((self.f1w,torch.zeros(2,1,device=tok.device)),1)
        out=torch.cat((self.outw,torch.zeros(9,1,device=tok.device)),1)
        return tok,pos,k,v,f1,out
    def forward(self,t):
        tok,pos,k,v,f1,out=self._expand(); x=F.embedding(t,tok)+pos[:t.shape[1]]
        f2=torch.cat((self.f2,torch.zeros(1,2,device=x.device)),0)
        for qfree,qb,ofree in ((self.q0,self.qb0,self.o0),(self.q1,self.qb1,self.o1)):
            o=torch.cat((ofree,torch.zeros(1,20,device=x.device)),0)
            q=torch.cat((qfree,torch.zeros(20,1,device=x.device)),1)
            z=F.layer_norm(x,(20,)); B,T,_=z.shape
            qq=F.linear(z,q,qb).view(B,T,4,5).transpose(1,2)
            kk=F.linear(z,k).view(B,1,T,5); vv=F.linear(z,v,self.vb).view(B,1,T,5)
            y=F.scaled_dot_product_attention(qq,kk,vv,is_causal=True)
            x=x+F.linear(y.transpose(1,2).reshape(B,T,20),o)
            x=x+F.linear(F.gelu(F.linear(F.layer_norm(x,(20,)),f1,self.f1b)),f2)
        nine=F.linear(F.layer_norm(x,(20,)),out,self.outb)
        return torch.cat((nine,torch.zeros((*nine.shape[:-1],1),device=nine.device)),2)
'''
TAIL='''
def build_model():
    model=AdditionTransformer(); values=torch.tensor(_VALUES); offset=0
    with torch.no_grad():
        for name,shape in _SHAPES.items():
            n=1
            for size in shape: n*=size
            getattr(model,name).copy_(values[offset:offset+n].reshape(shape)); offset+=n
    model.eval(); return model, {'architecture':'two-layer causal grouped-query transformer','trained':True,'digits':'least-significant-first'}
def add(model,a:int,b:int)->int:
    av=[int(c) for c in str(a)[::-1]]; bv=[int(c) for c in str(b)[::-1]]; seq=[]
    for x,y in zip(av,bv): seq.extend((x,y))
    seq.append(10); result=[]; device=next(model.parameters()).device
    with torch.no_grad():
        for _ in range(9):
            digit=int(model(torch.tensor([seq],device=device)) [0,-1].argmax())
            result.append(digit); seq.append(digit)
    return int(''.join(str(d) for d in result[::-1]))
'''
export()
