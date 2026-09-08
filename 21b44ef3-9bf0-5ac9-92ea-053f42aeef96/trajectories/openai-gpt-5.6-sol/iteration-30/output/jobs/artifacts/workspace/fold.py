import sys, torch
from pathlib import Path
sys.path.insert(0,'/workspace')
import train, submission
old=train.Model(2).cuda(); old.load_state_dict(torch.load('/workspace/model_w2.pt',map_location='cuda')); old.eval()
new=submission.AdditionTransformer().cuda(); new.eval()
os=dict(old.named_parameters()); ns=dict(new.named_parameters())
with torch.no_grad():
    for name,p in ns.items():
        if name=='ff1.weight': p.copy_(os[name]*os['norm_ff.weight'][None,:])
        elif name=='ff1.bias': p.copy_(os['ff1.weight']@os['norm_ff.bias'])
        elif name=='classifier.weight': p.copy_(os[name]*os['final_norm.weight'][None,:])
        elif name=='classifier.bias': p.copy_(os['classifier.weight']@os['final_norm.bias'])
        else: p.copy_(os[name])
    t=torch.randint(0,11,(2048,25),device='cuda')
    delta=(old(t)-new(t)).abs().max().item()
print('parameters',sum(p.numel() for p in new.parameters()),'max logit delta',delta)
print('folded eval',train.eval_model(new,500000,0),train.eval_model(new,500000,.7))
template=Path('/workspace/submission.py').read_text(); marker='_TRAINED = '
start=template.index(marker)+len(marker); end=template.index('\n\n\ndef build_model',start)
arrays=[p.detach().float().cpu().reshape(-1).tolist() for p in new.parameters()]
Path('/workspace/submission.py').write_text(template[:start]+repr(arrays)+template[end:])
torch.save(new.state_dict(),'/workspace/model_folded_2442.pt')
