import sys, time, torch
import torch.nn.functional as F
from submission import AdditionTransformer
from train import make_batch, evaluate

source = torch.load('/workspace/width4.pt', map_location='cpu', weights_only=True)['model']
importance = source['ff1.weight'].norm(dim=1) * source['ff2.weight'].norm(dim=0)
drop = int(importance.argmin())
keep = [i for i in range(4) if i != drop]
state = {k: v for k, v in source.items()}
state['ff1.weight'] = source['ff1.weight'][keep]
state['ff2.weight'] = source['ff2.weight'][:, keep]
model = AdditionTransformer().cuda()
model.load_state_dict(state)
print('drop', drop, 'importance', importance.tolist(), 'params', sum(p.numel() for p in model.parameters()), flush=True)
print('initial', evaluate(model, 10, 0.0), evaluate(model, 10, 0.8), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9,0.98), weight_decay=0.001)
best = -1
start=time.time()
for step in range(1, 30001):
    model.train()
    # Begin balanced, progressively emphasize broad structured cases.
    frac = 0.45 if step < 12000 else 0.60
    x,y=make_batch(8192, frac)
    logits=model(x)[:,16:25]
    loss=F.cross_entropy(logits.reshape(-1,10),y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step == 12000:
        for g in opt.param_groups: g['lr']=1e-5
    if step == 22000:
        for g in opt.param_groups: g['lr']=5e-6
    if step % 1000 == 0:
        a,ma=evaluate(model,8,0.0); e,me=evaluate(model,8,0.8)
        print(step, f'loss={loss.item():.6f} random={a:.6f} edge={e:.6f} margins={ma:.2f}/{me:.2f} sec={time.time()-start:.0f}',flush=True)
        score=min(a,e)
        if score>best:
            best=score; torch.save({'model':model.state_dict(),'step':step},'/workspace/width3_best.pt')
    if step % 5000 == 0: torch.save({'model':model.state_dict(),'step':step},'/workspace/width3_latest.pt')
