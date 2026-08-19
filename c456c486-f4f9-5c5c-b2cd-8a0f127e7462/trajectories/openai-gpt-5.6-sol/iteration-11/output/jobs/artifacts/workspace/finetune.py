import random
import time
import torch
import torch.nn.functional as F
from train import Adder, random_batch, carry_batch, outputs, DEVICE


def pattern_batch(batch):
    left = torch.zeros(batch, 15, dtype=torch.long, device=DEVICE)
    right = torch.zeros_like(left)
    kind = torch.randint(4, (batch,), device=DEVICE)
    da = torch.randint(10, (batch,), device=DEVICE)
    db = torch.randint(10, (batch,), device=DEVICE)
    # Repeated and complementary repeated pairs.
    db = torch.where(kind.eq(1), 9 - da, db)
    left[:, :14] = da[:, None]
    right[:, :14] = db[:, None]
    # Piecewise-constant blocks and sparse long chains.
    for p in range(14):
        change = torch.rand(batch, device=DEVICE) < 0.15
        da = torch.where(change, torch.randint(10, (batch,), device=DEVICE), da)
        db = torch.where(change, torch.randint(10, (batch,), device=DEVICE), db)
        mask = kind.ge(2)
        left[:, p] = torch.where(mask, da, left[:, p])
        right[:, p] = torch.where(mask, db, right[:, p])
    return left, right, outputs(left, right)


def explicit_data():
    pairs=[]; M=10**14-1
    for da in range(10):
      for db in range(10): pairs.append((int(str(da)*14),int(str(db)*14)))
    for start in range(14):
      for length in range(1,15-start):
        base=10**start; chain=(10**length-1)*base
        pairs.extend([(chain,base),(M-chain,chain),(10**(start+length)-base,base)])
    pairs += [(0,0),(M,0),(M,M),(M,1),(1,M),(99999999999990,9),(50000000000000,50000000000000)]
    l=torch.zeros(len(pairs),15,dtype=torch.long,device=DEVICE); r=torch.zeros_like(l)
    for j,(a,b) in enumerate(pairs):
      for p in range(14): l[j,p]=a%10;r[j,p]=b%10;a//=10;b//=10
    return l,r,outputs(l,r)

@torch.no_grad()
def eval_model(model, random_n=262144):
    model.eval(); wrong=0
    for _ in range(random_n//32768):
      l,r,t=random_batch(32768); wrong += model(l,r).argmax(-1).ne(t).any(1).sum().item()
    l,r,t=explicit_data(); ew=model(l,r).argmax(-1).ne(t).any(1).sum().item()
    l,r,t=carry_batch(65536); cw=model(l,r).argmax(-1).ne(t).any(1).sum().item()
    model.train(); return wrong,ew,cw


def main():
    ckpt=torch.load('/workspace/best.pt',weights_only=True)
    model=Adder(*ckpt['config']).to(DEVICE); model.load_state_dict(ckpt['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.98),weight_decay=.001)
    batch=4096; best=(10**9,None)
    print('start',ckpt['config'],eval_model(model),flush=True)
    for step in range(1,3001):
      l,r,t=random_batch(batch)
      # 15% arbitrary carry trajectories, 10% repeated/blockwise patterns.
      ncarry=batch*15//100; npat=batch*10//100
      cl,cr,ct=carry_batch(ncarry); pl,pr,pt=pattern_batch(npat)
      l[:ncarry],r[:ncarry],t[:ncarry]=cl,cr,ct
      l[ncarry:ncarry+npat],r[ncarry:ncarry+npat],t[ncarry:ncarry+npat]=pl,pr,pt
      loss=F.cross_entropy(model(l,r).flatten(0,1),t.flatten())
      opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
      if step%250==0:
        metrics=eval_model(model)
        # Edge failures strongly weighted, but never tolerate >0.02% random errors.
        score=metrics[0]+metrics[1]*500+metrics[2]*4
        print(step,float(loss),metrics,flush=True)
        if metrics[0] <= 52 and score < best[0]:
          best=(score,metrics)
          torch.save({'model':model.state_dict(),'config':ckpt['config'],'step':step,'metric':metrics},'/workspace/best_finetuned.pt')
    print('best',best,flush=True)

if __name__=='__main__': main()
