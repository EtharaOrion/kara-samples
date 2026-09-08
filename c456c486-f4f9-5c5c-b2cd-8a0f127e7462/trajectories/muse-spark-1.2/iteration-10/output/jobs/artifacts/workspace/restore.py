import torch, math
import torch.nn as nn, torch.nn.functional as F
D_MODEL=10; NHEAD=2; D_FF=20; VOCAB=12; SEQ_LEN=44; INPUT_LEN=29; OUTPUT_LEN=15
def _sinusoidal_pe(seq_len,d_model):
    pe=torch.zeros(seq_len,d_model)
    pos=torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*(-torch.log(torch.tensor(10000.0))/d_model))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div[:d_model//2])
    return pe
class TinyTransformer(nn.Module):
    def __init__(self,d_model=D_MODEL,nhead=NHEAD,d_ff=D_FF,vocab=VOCAB,seq_len=SEQ_LEN):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab,d_model)
        pe=_sinusoidal_pe(seq_len,d_model)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)
        self.ln_f=nn.LayerNorm(d_model)
        self.ff1=nn.Linear(d_model,d_ff)
        self.ff2=nn.Linear(d_ff,d_model)
        self.head=nn.Linear(d_model,vocab)
        mask=torch.triu(torch.full((seq_len,seq_len),float('-inf')),diagonal=1)
        self.register_buffer('causal_mask',mask)
    def forward(self,x):
        B,S=x.shape
        h=self.tok_emb(x)+self.pos_emb[:S].unsqueeze(0)
        h_norm=self.ln1(h)
        attn_out,_=self.attn(h_norm,h_norm,h_norm,attn_mask=self.causal_mask[:S,:S])
        h=h+attn_out
        h_norm2=self.ln2(h)
        ff=self.ff2(F.gelu(self.ff1(h_norm2)))
        h=h+ff
        h=self.ln_f(h)
        return self.head(h)

import os
# try best_final.pt
for path in ["/workspace/best_final.pt","/workspace/best_big.pt","/workspace/best_fast.pt","/workspace/best2.pt","/workspace/best.pt"]:
    if os.path.exists(path):
        print(path, os.path.getsize(path))

sd=torch.load("/workspace/best_final.pt", map_location="cpu")
# sd is dict of tensors
# convert to list form
code=f'''import torch
import torch.nn as nn
import torch.nn.functional as F
VOCAB=12;SEQ_LEN=44;INPUT_LEN=29;OUTPUT_LEN=15;D_MODEL={D_MODEL};NHEAD={NHEAD};D_FF={D_FF}
def _sinusoidal_pe(seq_len,d_model):
    pe=torch.zeros(seq_len,d_model)
    pos=torch.arange(seq_len,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*(-torch.log(torch.tensor(10000.0))/d_model))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div[:d_model//2])
    return pe
class TinyTransformer(nn.Module):
    def __init__(self,d_model=D_MODEL,nhead=NHEAD,d_ff=D_FF,vocab=VOCAB,seq_len=SEQ_LEN):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab,d_model)
        pe=_sinusoidal_pe(seq_len,d_model)
        self.register_buffer('pos_emb',pe)
        self.attn=nn.MultiheadAttention(d_model,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d_model)
        self.ln2=nn.LayerNorm(d_model)
        self.ln_f=nn.LayerNorm(d_model)
        self.ff1=nn.Linear(d_model,d_ff)
        self.ff2=nn.Linear(d_ff,d_model)
        self.head=nn.Linear(d_model,vocab)
        mask=torch.triu(torch.full((seq_len,seq_len),float('-inf')),diagonal=1)
        self.register_buffer('causal_mask',mask)
    def forward(self,x):
        B,S=x.shape
        h=self.tok_emb(x)+self.pos_emb[:S].unsqueeze(0)
        h_norm=self.ln1(h)
        attn_out,_=self.attn(h_norm,h_norm,h_norm,attn_mask=self.causal_mask[:S,:S])
        h=h+attn_out
        h_norm2=self.ln2(h)
        ff=self.ff2(F.gelu(self.ff1(h_norm2)))
        h=h+ff
        h=self.ln_f(h)
        return self.head(h)
_WEIGHTS={{}}
'''
for k,v in sd.items():
    if k in ('pos_emb','causal_mask'): continue
    code+=f"_WEIGHTS['{k}']={repr(v.tolist())}\n"
code+='''
def build_model():
    model=TinyTransformer()
    sd=model.state_dict()
    for k,lst in _WEIGHTS.items():
        sd[k]=torch.tensor(lst,dtype=sd[k].dtype)
    model.load_state_dict(sd)
    return model,{"vocab":VOCAB,"seq_len":SEQ_LEN,"d_model":D_MODEL,"nhead":NHEAD,"d_ff":D_FF}
def _encode(a,b):
    t=[]
    for i in range(14): t.append((a//(10**i))%10)
    t.append(10)
    for i in range(14): t.append((b//(10**i))%10)
    return t
def _decode(tokens):
    v=0
    for i,d in enumerate(tokens): v+=int(d)*(10**i)
    return v
def add(model,a,b):
    model.eval()
    with torch.no_grad():
        inp=_encode(a,b)
        seq=torch.tensor([inp],dtype=torch.long)
        gen=[]
        for step in range(OUTPUT_LEN):
            logits=model(seq)
            nxt_logit=logits[0,-1,:]
            nxt_logit[10]=float('-inf'); nxt_logit[11]=float('-inf')
            nxt=int(torch.argmax(nxt_logit).item())
            gen.append(nxt)
            seq=torch.cat([seq,torch.tensor([[nxt]],dtype=torch.long)],dim=1)
        return _decode(gen)
'''
open('/workspace/submission.py','w').write(code)
print("restored submission.py from best_final.pt (93%)")
import importlib.util
spec=importlib.util.spec_from_file_location("sub","/workspace/submission.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m,_=mod.build_model()
MAX_VAL=99999999999999
for av,bv in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876),(50000000000000,50000000000000)]:
    print(av,bv,mod.add(m,av,bv),av+bv,mod.add(m,av,bv)==av+bv)
print(f"params {sum(p.numel() for p in m.parameters())}")
