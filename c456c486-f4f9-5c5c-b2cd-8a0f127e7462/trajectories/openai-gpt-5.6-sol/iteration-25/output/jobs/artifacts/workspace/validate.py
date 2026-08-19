import sys, time, torch
sys.path.insert(0,'/workspace')
from model import Adder
from train import batch_data, answers

torch.backends.cuda.matmul.allow_tf32=True
model=Adder().cuda().eval(); model.load_state_dict(torch.load('/workspace/model_final.pt',weights_only=True))

def decode(a,b):
    prev=torch.empty((a.shape[0],0),dtype=torch.long,device='cuda')
    with torch.no_grad():
        for _ in range(15):
            prev=torch.cat((prev,model(a,b,prev)[:,-1].argmax(-1,keepdim=True)),1)
    return prev

def random_test(total, structured):
    errors=0; seen=0
    while seen<total:
        n=min(8192,total-seen); a,b,y=batch_data(n,structured)
        errors += (decode(a,b)!=y).any(1).sum().item(); seen += n
    return errors

def to_digits(values):
    values=torch.tensor(values,dtype=torch.long,device='cuda'); out=torch.zeros((len(values),15),dtype=torch.long,device='cuda')
    for j in range(15): out[:,j]=(values//(10**j))%10
    return out

cases={(0,0),(1,1),(99999999999999,1),(99999999999999,99999999999999),(50000000000000,50000000000000)}
P=10**14
for length in range(1,15):
    run=10**length-1
    for start in range(0,15-length):
        scale=10**start
        for delta in (-2,-1,0,1,2):
            x=run*scale+delta*scale
            if 0<=x<P:
                for y in (1*scale,2*scale,7*scale,P-1-x if x<P else 0):
                    if 0<=y<P: cases.add((x,y)); cases.add((y,x))
for k in range(15):
    p=10**k
    for d in range(-20,21):
        x=p+d
        if 0<=x<P:
            for y in (0,1,9,10,99,99999999999999-x):
                if 0<=y<P: cases.add((x,y)); cases.add((y,x))
pairs=list(cases); a=to_digits([x for x,y in pairs]); b=to_digits([y for x,y in pairs]); y=answers(a,b)
t=time.time()
print('systematic',int((decode(a,b)!=y).any(1).sum()),len(pairs),'seconds',time.time()-t,flush=True)
for structured in (False,True):
    t=time.time(); e=random_test(1048576,structured); print('random',structured,e,1048576,'seconds',time.time()-t,flush=True)
# Attention content dependence: compare normalized QK scores for two source sequences.
a,b,_=batch_data(2,False); source=model.a_embed(a)+model.b_embed(b); x=source+model.position[:15]
z=model.blocks[0].norm1(x); q,k,_=model.blocks[0].qkv(z).chunk(3,-1)
s=torch.matmul(q.view(2,15,2,5).transpose(1,2),k.view(2,15,2,5).transpose(1,2).transpose(-2,-1))
print('qk_input_delta',float((s[0]-s[1]).abs().max()))
print('parameters',sum(p.numel() for p in model.parameters()))
