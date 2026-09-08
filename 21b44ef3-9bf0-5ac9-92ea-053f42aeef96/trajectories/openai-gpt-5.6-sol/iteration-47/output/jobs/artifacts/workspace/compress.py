import itertools, math, torch
from pathlib import Path
S=torch.load('/workspace/final.pt',map_location='cpu',weights_only=True)['model']
# Fold attention LayerNorm scale, retaining normalized beta.
g=S['norm_attn.weight']; S['norm_attn.bias']=S['norm_attn.bias']/g
for n in ['key.weight','value.weight','query.0.weight','query.1.weight']:
    S[n]=S[n]*g[None,:]
# Fold FF and output LayerNorm affine maps.
g,b=S['norm_ff.weight'],S['norm_ff.bias']; W=S['ff1.weight']; S['ff1.weight']=W*g; S['ff1.bias']=W@b
g,b=S['norm_out.weight'],S['norm_out.bias']; W=S['classifier.weight']; S['classifier.weight']=W*g; S['classifier.bias']=W@b
# Residual-stream all-ones gauges.
S['token.weight']=S['token.weight']-S['token.weight'][:,-1:]
R=S['pos_right']; S['pos_right']=R-R[:,-1:]
for n in ['ff2.weight','project.0.weight','project.1.weight']:
    W=S[n]; S[n]=W-W[-1:,:]
# Position GL(2) gauge: choose best-conditioned pair of rows.
L,R=S['pos_left'],S['pos_right']
best=max(itertools.combinations(range(25),2),key=lambda ij: abs(float(torch.linalg.det(L[list(ij)]))))
B=L[list(best)]; S['pos_left']=L@torch.linalg.inv(B); S['pos_right']=B@R
# Independent K and V GL(5) gauges.
def bestcols(W):
    return max(itertools.combinations(range(20),5),key=lambda js: abs(float(torch.linalg.det(W[:,list(js)]))))
K=S['key.weight']
best_key=max(((js,z) for js in itertools.combinations(range(20),5) for z in range(20) if z not in js), key=lambda jz: abs(float(torch.linalg.det(K[:,list(jz[0])]-K[:,jz[1]:jz[1]+1]))))
kj,kzero=best_key; u=K[:,kzero:kzero+1]; B=K[:,list(kj)]-u
S['key.weight']=torch.linalg.solve(B,K-u)
for i in range(2):
    W=S[f'query.{i}.weight'].reshape(4,5,20)
    S[f'query.{i}.weight']=torch.einsum('ji,hjk->hik',B,W).reshape(20,20)
vj=bestcols(S['value.weight']); C=S['value.weight'][:,list(vj)]
S['value.weight']=torch.linalg.solve(C,S['value.weight'])
block=torch.block_diag(C,C,C,C)
for i in range(2): S[f'project.{i}.weight']=S[f'project.{i}.weight']@block
# Project gauges must be applied after the V transformation.
for n in ['project.0.weight','project.1.weight']:
    W=S[n]; S[n]=W-W[-1:,:]
# Non-affine LayerNorm inputs have zero coordinate sum.
W=S['ff1.weight']; S['ff1.weight']=W-W[:,-1:]
# Reference logit 9, then classifier input null gauge.
S['classifier.weight']=S['classifier.weight'][:9]-S['classifier.weight'][9:10]
S['classifier.bias']=S['classifier.bias'][:9]-S['classifier.bias'][9]
W=S['classifier.weight']; S['classifier.weight']=W-W[:,-1:]

pos_keep=[i for i in range(25) if i not in best]
kkeep=[i for i in range(20) if i not in kj and i != kzero]
vkeep=[i for i in range(20) if i not in vj]
P={
 'token':S['token.weight'][:,:19], 'pos_free':S['pos_left'][pos_keep], 'pos_right':S['pos_right'][:,:19],
 'norm_bias':S['norm_attn.bias'], 'key_free':S['key.weight'][:,kkeep], 'value_free':S['value.weight'][:,vkeep],
 'q0':S['query.0.weight'],'q1':S['query.1.weight'],
 'p0':S['project.0.weight'][:19], 'p1':S['project.1.weight'][:19],
 'ff':S['ff1.weight'][:,:19], 'ff1b':S['ff1.bias'], 'ff2':S['ff2.weight'][:19],
 'classifier':S['classifier.weight'][:,:19], 'classifierb':S['classifier.bias']}

def lit(x): return f'torch.tensor({x.tolist()!r})'
source='''import math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nD=20\nH=4\nHD=5\n_POS_FIXED=%r\n_POS_KEEP=%r\n_K_FIXED=%r\n_K_ZERO=%r\n_K_KEEP=%r\n_V_FIXED=%r\n_V_KEEP=%r\n\nclass AdditionTransformer(nn.Module):\n    def __init__(self):\n        super().__init__()\n'''%(best,pos_keep,kj,kzero,kkeep,vj,vkeep)
for name,val in P.items(): source+=f"        self.{name}=nn.Parameter({lit(val)})\n"
source+='''\n    def forward(self,tokens):
        length=tokens.shape[1]
        token=torch.cat((self.token,self.token.new_zeros(11,1)),1)
        left=self.pos_free.new_zeros(25,2)
        left[list(_POS_FIXED)]=torch.eye(2,device=left.device,dtype=left.dtype)
        left[_POS_KEEP]=self.pos_free
        right=torch.cat((self.pos_right,self.pos_right.new_zeros(2,1)),1)
        x=F.embedding(tokens,token)+(left[:length]@right)
        mask=torch.ones(length,length,device=tokens.device,dtype=torch.bool).triu(1)
        key=self.key_free.new_zeros(5,20); key[:,_K_FIXED]=torch.eye(5,device=key.device,dtype=key.dtype); key[:,_K_ZERO]=0; key[:,_K_KEEP]=self.key_free
        value=self.value_free.new_zeros(5,20); value[:,_V_FIXED]=torch.eye(5,device=value.device,dtype=value.dtype); value[:,_V_KEEP]=self.value_free
        ff1=torch.cat((self.ff,self.ff.new_zeros(2,1)),1)
        ff2=torch.cat((self.ff2,self.ff2.new_zeros(1,2)),0)
        for q,p in ((self.q0,self.p0),(self.q1,self.p1)):
            z=F.layer_norm(x,(20,))+self.norm_bias
            query=F.linear(z,q).view(z.shape[0],length,4,5).transpose(1,2)
            k=F.linear(z,key); v=F.linear(z,value)
            scores=torch.einsum('bhtd,bsd->bhts',query,k)/math.sqrt(5)
            weights=torch.softmax(scores.masked_fill(mask,float('-inf')),dim=-1)
            attended=torch.einsum('bhts,bsd->bhtd',weights,v).transpose(1,2).reshape(z.shape[0],length,20)
            project=torch.cat((p,p.new_zeros(1,20)),0)
            x=x+F.linear(attended,project)
            x=x+F.linear(F.gelu(F.linear(F.layer_norm(x,(20,)),ff1,self.ff1b)),ff2)
        classifier=torch.cat((self.classifier,self.classifier.new_zeros(9,1)),1)
        logits=F.linear(F.layer_norm(x,(20,)),classifier,self.classifierb)
        return torch.cat((logits,logits.new_zeros(*logits.shape[:-1],1)),dim=-1)


def build_model():
    model=AdditionTransformer()
    model.eval()
    return model,{"task":"8-digit addition","architecture":"causal grouped-query transformer"}


def add(model,a:int,b:int)->int:
    ad=[ord(c)-48 for c in str(a)[::-1]]
    bd=[ord(c)-48 for c in str(b)[::-1]]
    prefix=[]
    for x,y in zip(ad,bd): prefix.extend((x,y))
    tokens=torch.tensor([prefix+[10]],dtype=torch.long,device=next(model.parameters()).device)
    out=[]
    with torch.no_grad():
        for _ in range(9):
            digit=int(model(tokens)[0,-1].argmax().item())
            out.append(digit)
            tokens=torch.cat((tokens,torch.tensor([[digit]],device=tokens.device)),1)
    return int(''.join(str(x) for x in out[::-1]))
'''
Path('/workspace/submission_compressed.py').write_text(source)
print('fixed',best,kj,kzero,vj)
