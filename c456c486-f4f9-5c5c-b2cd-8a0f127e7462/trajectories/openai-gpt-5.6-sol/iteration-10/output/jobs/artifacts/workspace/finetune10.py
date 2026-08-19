import sys
if not hasattr(sys, "get_int_max_str_digits"):
    def _get_digits() -> int: return 4300
    def _set_digits(maxdigits: int) -> None: return None
    sys.get_int_max_str_digits = _get_digits
    sys.set_int_max_str_digits = _set_digits
import torch
import torch.nn.functional as F
from submission import AdderTransformer
from train import make_batch, labels_for, random_validation, edge_validation, exact_accuracy, export

torch.manual_seed(55419)
torch.set_num_threads(13)
model = AdderTransformer()
model.load_state_dict(torch.load('/workspace/best10.pt', weights_only=True))
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.001)
rv = random_validation(30000, 772211)
ev = edge_validation()
best_score = (exact_accuracy(model, ev), exact_accuracy(model, rv))
best = {k:v.clone() for k,v in model.state_dict().items()}
print('initial', best_score, flush=True)
for step in range(1, 1601):
    a,b,y = make_batch(2048, .30, 'cpu')
    n=512
    digits=torch.randint(0,10,(n,1))
    repeated=digits.expand(n,14)
    a[:n,:14]=repeated
    modes=torch.arange(n).remainder(3)
    b[:n,:14]=torch.where((modes==0).view(-1,1), repeated, torch.where((modes==1).view(-1,1), 9-repeated, torch.zeros_like(repeated)))
    a[:n,-1]=0; b[:n,-1]=0
    y=labels_for(a,b)
    loss=F.cross_entropy(model(a,b).reshape(-1,10),y.reshape(-1))
    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
    if step==1200:
        optimizer.param_groups[0]['lr']=3e-5
    if step%100==0:
        ea=exact_accuracy(model,ev); ra=exact_accuracy(model,rv); score=(ea,ra)
        if score>best_score:
            best_score=score; best={k:v.detach().clone() for k,v in model.state_dict().items()}; torch.save(best,'/workspace/best10.pt'); export(model,'/workspace/submission.py')
        print(step,loss.item(),score,'best',best_score,flush=True)
model.load_state_dict(best); export(model,'/workspace/submission.py')
