from pathlib import Path
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace')
import train
import finetune


def sparse_pair(n, device):
    k = torch.randint(0, 14, (n,), device=device)
    p = torch.pow(torch.tensor(10, dtype=torch.int64, device=device), k)
    da = torch.randint(0, 10, (n,), device=device)
    db = torch.randint(0, 10, (n,), device=device)
    # Optional independent low tails force sparse-column behavior in varied contexts.
    tail_a = (torch.rand(n, device=device) * p.float()).long()
    tail_b = (torch.rand(n, device=device) * p.float()).long()
    use_tail = torch.rand(n, device=device) < 0.35
    a = da * p + torch.where(use_tail, tail_a, 0)
    b = db * p + torch.where(use_tail, tail_b, 0)
    return train.to_digits(a), train.to_digits(b), train.to_digits(a + b)


def main():
    torch.manual_seed(20250816)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    train.a_device = device
    model = train.Adder().to(device)
    model.load_state_dict(torch.load('/workspace/best.pt', weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0005)
    for step in range(1, 3001):
        x1,b1,y1 = train.make_batch(4096, device, 0.4)
        x2,b2,y2 = finetune.boundary_batch(2048, device)
        x3,b3,y3 = sparse_pair(2048, device)
        a = torch.cat((x1,x2,x3)); b = torch.cat((b1,b2,b3)); y = torch.cat((y1,y2,y3))
        loss = F.cross_entropy(model(a,b).reshape(-1,10), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            er,nr=train.evaluate(model,16,4096,0.0); es,ns=train.evaluate(model,8,4096,0.75)
            with torch.no_grad():
                x,z,yy=sparse_pair(32768,device); ep=(model(x,z).argmax(-1)!=yy).any(1).sum().item()
            print(step,float(loss),'random',er,nr,'structured',es,ns,'sparse',ep,32768,flush=True)
    torch.save(model.state_dict(), '/workspace/best.pt')
    train.export(model, Path('/workspace/submission.py'))
if __name__ == '__main__': main()
