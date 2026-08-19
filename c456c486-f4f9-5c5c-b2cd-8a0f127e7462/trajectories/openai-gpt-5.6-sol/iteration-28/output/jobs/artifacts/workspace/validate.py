import sys
sys.path.insert(0, '/workspace')
import torch
import train

m=train.Model().cuda()
m.load_state_dict(torch.load('/workspace/final.pt', weights_only=True))
print('uniform_1048576', train.greedy_errors(m, 1048576, False), flush=True)
print('structured_1048576', train.greedy_errors(m, 1048576, True), flush=True)

cases={(0,0),(1,0),(0,1),(99_999_999_999_999,0),(99_999_999_999_999,1),(99_999_999_999_999,99_999_999_999_999)}
for p in range(14):
    q=10**p
    for x in range(1,10):
        cases.add((x,q))
        cases.add((x,q+x))
        cases.add((x*q,x*q))
        cases.add((q-1,1))
        cases.add((q-1,q+1))
        cases.add((5,q+5))
        cases.add((q*10-1,1))
        cases.add((q*10-1,q*10-1))
for length in range(1,15):
    run=10**length-1
    for start in range(15-length):
        scale=10**start
        cases.add((run*scale,scale))
        cases.add((run*scale,run*scale))
        cases.add((run*scale,MAX:=99_999_999_999_999-run*scale))
cases=[x for x in cases if 0<=x[0]<train.MAX_N and 0<=x[1]<train.MAX_N]
@torch.no_grad()
def check(cases):
    bad=[]
    for off in range(0,len(cases),4096):
        chunk=cases[off:off+4096]
        a=torch.tensor([x[0] for x in chunk],device='cuda')
        b=torch.tensor([x[1] for x in chunk],device='cuda')
        ad,bd,target=train.to_digits(a),train.to_digits(b),train.to_digits(a+b)
        prev=torch.empty((len(chunk),0),dtype=torch.long,device='cuda')
        pred=[]
        for _ in range(15):
            d=m(ad,bd,prev)[:,-1].argmax(1); pred.append(d); prev=torch.cat((prev,d[:,None]),1)
        pi=train.digits_to_int(torch.stack(pred,1))
        for i in (pi != a+b).nonzero().flatten().tolist(): bad.append((chunk[i],pi[i].item(),(a+b)[i].item()))
    return bad
bad=check(cases)
print('systematic',len(bad),'/',len(cases),bad[:20],flush=True)
# Attention content dependence through first block's actual normalized Q/K scores.
ad1,bd1=train.to_digits(torch.tensor([12345678901234],device='cuda')),train.to_digits(torch.tensor([9876543210987],device='cuda'))
ad2,bd2=train.to_digits(torch.tensor([11111111111111],device='cuda')),train.to_digits(torch.tensor([22222222222222],device='cuda'))
b=m.blocks[0]
x1=m.a_emb(ad1)+m.b_emb(bd1)+m.pos[:15]
x2=m.a_emb(ad2)+m.b_emb(bd2)+m.pos[:15]
q1,k1,_=b.qkv(b.ln1(x1)).chunk(3,-1); q2,k2,_=b.qkv(b.ln1(x2)).chunk(3,-1)
s1=q1.view(1,15,2,5).transpose(1,2)@k1.view(1,15,2,5).transpose(1,2).transpose(-1,-2)
s2=q2.view(1,15,2,5).transpose(1,2)@k2.view(1,15,2,5).transpose(1,2).transpose(-1,-2)
print('qk_max_change',float((s1-s2).abs().max()),flush=True)
