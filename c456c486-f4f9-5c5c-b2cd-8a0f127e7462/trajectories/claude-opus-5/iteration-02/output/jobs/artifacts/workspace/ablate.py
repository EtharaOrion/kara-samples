"""Check that the attention pattern genuinely depends on the input."""
import sys, torch
from data import make_batch
from model_def import TinyAdder
from train import evaluate

def report(m, dev='cpu', n=200000):
    base = evaluate(m, dev, n=n, bs=100000)
    wq = m.wq.detach().clone(); bq = m.bq.detach().clone()
    wk = m.wk.detach().clone(); bk = m.bk.detach().clone()
    with torch.no_grad():   # kill content-based scores, keep positional ALiBi
        m.wq.zero_(); m.bq.zero_(); m.wk.zero_(); m.bk.zero_()
    pos_only = evaluate(m, dev, n=n, bs=100000)
    with torch.no_grad():
        m.wq.copy_(wq); m.bq.copy_(bq); m.wk.copy_(wk); m.bk.copy_(bk)
    da, db, _ = make_batch(4096, dev)
    m(da, db); a = m.attn[:, 1:]
    var = a.var(0).mean().item()          # variance of attention across inputs
    ent = -(a.clamp_min(1e-9).log() * a).sum(-1).mean().item()
    print(f"accuracy {base:.5f} | positional-only attention {pos_only:.5f} "
          f"| attn var across inputs {var:.4f} | entropy {ent:.3f}")
    return base, pos_only

if __name__ == '__main__':
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    c = ck['config']
    m = TinyAdder(d_model=c['d_model'], d_ff=c['d_ff'], head=c['head'],
                  mlp_full_out=c['mlp_out']=='full', pair_mode=c.get('pair_mode','table'),
                  d_feat=c.get('d_feat',3))
    m.load_state_dict(ck['state'], strict=False)
    report(m)
