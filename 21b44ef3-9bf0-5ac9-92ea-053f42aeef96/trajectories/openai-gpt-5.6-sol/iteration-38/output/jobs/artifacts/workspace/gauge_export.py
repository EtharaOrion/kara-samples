import sys
from pathlib import Path
sys.path.append('/usr/local/lib/python3.11/dist-packages'); sys.path.insert(0,'/workspace')
import torch
from torch import nn
class A(nn.Module):
 def __init__(self):
  super().__init__(); self.q=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.o=nn.ModuleList([nn.Linear(20,20,bias=False) for _ in range(2)]); self.k=nn.Linear(20,5,bias=False); self.v=nn.Linear(20,5,bias=False)
 def forward(self,x,l):
  B,L,_=x.shape; q=self.q[l](x).view(B,L,4,5).transpose(1,2); k=self.k(x); v=self.v(x); y=torch.nn.functional.scaled_dot_product_attention(q,k[:,None],v[:,None],is_causal=True,enable_gqa=True); return self.o[l](y.transpose(1,2).reshape(B,L,20))
class AdditionTransformer(nn.Module):
 def __init__(self):
  super().__init__(); self.token=nn.Embedding(11,20); self.pos_a=nn.Parameter(torch.empty(25,2)); self.pos_b=nn.Parameter(torch.empty(2,20)); self.attn_norm=nn.LayerNorm(20); self.attn=A(); self.ff_norm=nn.LayerNorm(20); self.ff1=nn.Linear(20,2,bias=False); self.ff2=nn.Linear(2,20,bias=False); self.final_norm=nn.LayerNorm(20); self.head=nn.Linear(20,10,bias=False)
 def forward(self,t):
  x=self.token(t)+(self.pos_a@self.pos_b)[:t.shape[1]]
  for l in range(2): x=x+self.attn(self.attn_norm(x),l); x=x+self.ff2(torch.nn.functional.gelu(self.ff1(self.ff_norm(x))))
  return self.head(self.final_norm(x))
m=AdditionTransformer().cuda(); m.load_state_dict(torch.load('/workspace/final.pt',weights_only=True),strict=False); m.eval()
x=torch.randint(0,11,(256,25),device='cuda')
with torch.no_grad(): reference=m(x)

# Fold ff and final LayerNorm affines into their following linear maps.
with torch.no_grad():
    g,b=m.ff_norm.weight,m.ff_norm.bias; W=m.ff1.weight
    ff1_w=W*g; ff1_b=W@b
    g,b=m.final_norm.weight,m.final_norm.bias; W=m.head.weight
    head_w=W*g; head_b=W@b

    # GL(2): make positional rows 0 and 1 fixed identity.
    A=m.pos_a; B=m.pos_b; R=A[[0,1]]
    free_pos=A@torch.linalg.inv(R)
    pos_b=R@B

    # Independent GL(5): each K/V projection gets first five columns fixed identity.
    def kv_gauge(W):
        C=W[:,:5]
        G=torch.linalg.inv(C)
        return W[:,5:]@G, G.inverse()  # free 5x15, right factor applied to input below
    # Need W' = G W with first cols I; compensate q head by q_h' = G^{-T} q_h.
    def transform_k(W, qs):
        C=W[:,:5]; G=torch.linalg.inv(C); nw=G@W
        nqs=[]
        for q in qs:
            qh=q.view(4,5,20)
            nqs.append(torch.einsum('ij,hjk->hik',C.T,qh).reshape(20,20))
        return nw[:,5:],nqs
    def transform_v(W, os):
        C=W[:,:5]; G=torch.linalg.inv(C); nw=G@W
        nos=[]
        for o in os:
            oh=o.view(20,4,5)
            nos.append(torch.einsum('dhj,jk->dhk',oh,C).reshape(20,20))
        return nw[:,5:],nos
    k_free, qws=transform_k(m.attn.k.weight,[z.weight for z in m.attn.q])
    v_free, ows=transform_v(m.attn.v.weight,[z.weight for z in m.attn.o])

    # LayerNorm output has zero feature mean: set one input column per output row to zero.
    def null_gauge(W): return W[:,:-1]-W[:,-1:]
    ff1_free=null_gauge(ff1_w)

    # Final normalized representation has zero mean; then fix class 9 as logit reference.
    hw=null_gauge(head_w)
    head_free=hw[:9]-hw[9:10]; head_bias=(head_b[:9]-head_b[9]).contiguous()

# Validate transformations with a compact emulator before export.
def forward_new(t):
    L=t.shape[1]
    token=m.token.weight-m.token.weight[:,0:1]
    # Removing token dim-0 is a common translation, erased by attn_norm before every used branch.
    xx=token[t]+(free_pos@pos_b)[:L]
    I=torch.eye(5,device=t.device)
    kw=torch.cat((I,k_free),1); vw=torch.cat((I,v_free),1)
    for layer in range(2):
        z=m.attn_norm(xx); z19=z[...,:-1]
        q= torch.nn.functional.linear(z,qws[layer]).view(len(t),L,4,5).transpose(1,2)
        k=torch.nn.functional.linear(z,kw); v=torch.nn.functional.linear(z,vw)
        y=torch.nn.functional.scaled_dot_product_attention(q,k[:,None],v[:,None],is_causal=True,enable_gqa=True)
        xx=xx+torch.nn.functional.linear(y.transpose(1,2).reshape(len(t),L,20),ows[layer])
        z=torch.nn.functional.layer_norm(xx,(20,))
        xx=xx+m.ff2(torch.nn.functional.gelu(torch.nn.functional.linear(z[...,:-1],ff1_free,ff1_b)))
    z=torch.nn.functional.layer_norm(xx,(20,))
    nine=torch.nn.functional.linear(z[...,:-1],head_free,head_bias)
    return torch.cat((nine,torch.zeros_like(nine[...,:1])),-1)
with torch.no_grad():
    candidate=forward_new(x)
    print('max centered delta',float(((reference-reference[...,9:]) - candidate).abs().max()),'argmax',bool((reference.argmax(-1)==candidate.argmax(-1)).all()))

# Parameter order for compact final architecture.
vals=[
 (m.token.weight-m.token.weight[:,0:1])[:,1:].cpu(), free_pos[2:].cpu(), pos_b.cpu(),
 ff1_free.cpu(),ff1_b.cpu(),(m.ff2.weight-m.ff2.weight[:1])[1:].cpu(),head_free.cpu(),head_bias.cpu(),
 m.attn_norm.weight.cpu(),m.attn_norm.bias.cpu(),k_free.cpu(),v_free.cpu(),
 qws[0].cpu(),qws[1].cpu(),ows[0].cpu(),ows[1].cpu(),
]
state=[v.flatten().tolist() for v in vals]
Path('/workspace/gauged_state.txt').write_text(repr(state))
