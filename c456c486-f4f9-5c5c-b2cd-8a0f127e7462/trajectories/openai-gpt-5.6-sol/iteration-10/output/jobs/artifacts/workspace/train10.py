import sys
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_digits() -> int: return 4300
    def _set_digits(maxdigits: int) -> None: return None
    sys.get_int_max_str_digits = _get_digits
    sys.set_int_max_str_digits = _set_digits
import torch
import torch.nn.functional as F
from submission import AdderTransformer
from train import make_batch, random_validation, edge_validation, exact_accuracy

torch.manual_seed(19283); torch.set_num_threads(13)
m=AdderTransformer(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=.003)
rv=random_validation(30000,314159); ev=edge_validation(); best=(-1.,-1.); state=None
for step in range(1,6001):
    frac=.35 if step<3500 else .5
    a,b,y=make_batch(2048,frac,'cpu'); loss=F.cross_entropy(m(a,b).reshape(-1,10),y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    if step==3000: opt.param_groups[0]['lr']=1e-3
    if step==4500: opt.param_groups[0]['lr']=3e-4
    if step%250==0:
        ea=exact_accuracy(m,ev); ra=exact_accuracy(m,rv); score=(ea,ra)
        if score>best: best=score; state={k:v.detach().clone() for k,v in m.state_dict().items()}; torch.save(state,'/workspace/best10.pt')
        print(step,loss.item(),score,'best',best,flush=True)
print('parameters',sum(p.numel() for p in m.parameters()))
