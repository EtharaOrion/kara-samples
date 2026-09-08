import math
import os
import random
import sys
import time
import torch
from torch.nn import functional as F

sys.path.insert(0, "/workspace")
from submission import AdditionTransformer

DEVICE = "cuda"
BATCH = 8192
LOW = 10_000_000
HIGH = 99_999_999
LIMIT = 100_000_000


def make_pairs(n, structured=0.32):
    a = torch.randint(LOW, LIMIT, (n,), device=DEVICE)
    b = torch.randint(LOW, LIMIT, (n,), device=DEVICE)
    count = int(n * structured)
    if not count:
        return a, b
    x = torch.randint(LOW, LIMIT, (count,), device=DEVICE)
    mode = torch.randint(0, 8, (count,), device=DEVICE)
    y = torch.randint(LOW, LIMIT, (count,), device=DEVICE)

    # Complements and near-complements exercise complete carry propagation.
    m = mode == 0
    delta = torch.randint(-20, 21, (count,), device=DEVICE)
    y[m] = (LIMIT - x[m] + delta[m]).clamp(LOW, HIGH)

    # Every possible trailing carry-chain length, including asymmetric endings.
    for which in (1, 2):
        m = mode == which
        k = torch.randint(1, 8, (count,), device=DEVICE)
        p = 10 ** k
        suffix = x.remainder(p)
        jitter = torch.randint(-3, 4, (count,), device=DEVICE)
        yy = (p - suffix + jitter).remainder(p)
        prefix = torch.randint(1, 10, (count,), device=DEVICE) * (LIMIT // 10)
        cand = prefix + yy
        y[m] = cand[m].clamp(LOW, HIGH)

    # Decimal boundaries and sparse low-order digits.
    m = mode == 3
    powers = torch.tensor([10,100,1000,10000,100000,1000000,10000000], device=DEVICE)
    p = powers[torch.randint(0, 7, (count,), device=DEVICE)]
    y[m] = ((y[m] // p[m]) * p[m] + torch.randint(0, 3, (count,), device=DEVICE)[m]).clamp(LOW, HIGH)

    # Runs of nines or zeros with arbitrary high prefixes.
    m = mode == 4
    k = torch.randint(1, 8, (count,), device=DEVICE)
    p = 10 ** k
    high = (y // p) * p
    choose9 = torch.randint(0, 2, (count,), device=DEVICE).bool()
    tail = torch.where(choose9, p - 1, torch.zeros_like(p))
    y[m] = (high[m] + tail[m]).clamp(LOW, HIGH)

    # Repeated-digit operands.
    m = mode == 5
    d = torch.randint(1, 10, (count,), device=DEVICE)
    rep = d * 11_111_111
    y[m] = rep[m]

    # Near extrema.
    m = mode == 6
    side = torch.randint(0, 2, (count,), device=DEVICE).bool()
    edge = torch.where(side, LOW + torch.randint(0, 10000, (count,), device=DEVICE), HIGH - torch.randint(0, 10000, (count,), device=DEVICE))
    y[m] = edge[m]

    # Sparse interior decimal digit patterns.
    m = mode == 7
    p1 = 10 ** torch.randint(0, 8, (count,), device=DEVICE)
    p2 = 10 ** torch.randint(0, 8, (count,), device=DEVICE)
    sparse = torch.randint(1,10,(count,),device=DEVICE) * 10_000_000 + torch.randint(0,10,(count,),device=DEVICE)*p1 + torch.randint(0,10,(count,),device=DEVICE)*p2
    y[m] = sparse[m].clamp(LOW, HIGH)
    a[:count], b[:count] = x, y
    return a, b


def encode(a, b):
    n = a.numel()
    ad, bd = [], []
    aa, bb = a, b
    for _ in range(8):
        ad.append(aa.remainder(10)); bd.append(bb.remainder(10))
        aa = aa // 10; bb = bb // 10
    s = a + b
    out = []
    for _ in range(9):
        out.append(s.remainder(10)); s = s // 10
    digits = torch.stack([v for pair in zip(ad, bd) for v in pair], 1)
    target = torch.stack(out, 1)
    inp = torch.cat((digits, torch.full((n,1),10,device=DEVICE), target[:,:8]), 1)
    return inp.long(), target.long()


@torch.no_grad()
def exact_accuracy(model, n=100000, structured=0.0, chunk=10000):
    model.eval(); good = total = 0; min_margin = 1e9
    for _ in range((n + chunk - 1)//chunk):
        size = min(chunk, n-total)
        a,b = make_pairs(size, structured)
        base, target = encode(a,b)
        seq = base[:,:17]
        pred = []
        for j in range(9):
            logits = model(seq)[:,-1]
            top = logits.topk(2, dim=-1).values
            min_margin = min(min_margin, float((top[:,0]-top[:,1]).min()))
            d = logits.argmax(-1); pred.append(d)
            if j != 8: seq = torch.cat((seq,d[:,None]),1)
        good += int((torch.stack(pred,1)==target).all(1).sum())
        total += size
    model.train()
    return good/total, min_margin


def train_stage(model, steps, peak_lr, structured, name, warmup=300):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9,0.98), weight_decay=0.01)
    start = time.time()
    for step in range(1, steps+1):
        if step <= warmup:
            lr = peak_lr * step / warmup
        else:
            q = (step-warmup)/max(1,steps-warmup)
            lr = peak_lr * (0.08 + 0.92*0.5*(1+math.cos(math.pi*q)))
        for group in opt.param_groups: group["lr"] = lr
        a,b = make_pairs(BATCH, structured)
        inp,target = encode(a,b)
        logits = model(inp)[:,16:25]
        loss = F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == steps:
            acc, margin = exact_accuracy(model, 20000, 0.0)
            sacc, _ = exact_accuracy(model, 20000, 0.75)
            print(f"{name} {step}/{steps} loss={loss.item():.5f} acc={acc:.5f} struct={sacc:.5f} margin={margin:.3f} lr={lr:.2g} sec={time.time()-start:.0f}", flush=True)
            torch.save(model.state_dict(), f"/workspace/{name}.pt")
    return model


def prune(model, width):
    old = model.ff_in.out_features
    score = model.ff_in.weight.norm(dim=1) * model.ff_out.weight.norm(dim=0)
    keep = score.topk(width).indices.sort().values
    new = AdditionTransformer(width).to(DEVICE)
    state = model.state_dict()
    newstate = new.state_dict()
    for key in newstate:
        if key == "ff_in.weight": newstate[key] = state[key][keep]
        elif key == "ff_out.weight": newstate[key] = state[key][:,keep]
        else: newstate[key] = state[key]
    new.load_state_dict(newstate)
    print(f"pruned {old}->{width}; kept {keep.tolist()}", flush=True)
    return new


def export(model):
    model = model.float().cpu().eval()
    vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach().tolist()
    path = "/workspace/submission.py"
    text = open(path).read()
    start = text.index("# TRAINED_WEIGHTS_START")
    end = text.index("# TRAINED_WEIGHTS_END") + len("# TRAINED_WEIGHTS_END")
    vals = ",".join(format(v, ".9g") for v in vector)
    block = f"# TRAINED_WEIGHTS_START\n_TRAINED = [{vals}]\n# TRAINED_WEIGHTS_END"
    text = text[:start] + block + text[end:]
    text = text.replace("def build_model():\n    model = AdditionTransformer(4)", f"def build_model():\n    model = AdditionTransformer({model.ff_in.out_features})")
    temp = path + ".new"
    open(temp,"w").write(text); os.replace(temp,path)
    print(f"exported {len(vector)} parameters to {path}", flush=True)


def main():
    torch.manual_seed(2025); random.seed(2025)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AdditionTransformer(4).to(DEVICE)
    train_stage(model, 36000, 2e-3, 0.18, "teacher", 1000)
    print("teacher final", exact_accuracy(model,500000,0.0), exact_accuracy(model,500000,0.7), flush=True)
    export(model)
    model = prune(model,3)
    train_stage(model,18000,2e-5,0.42,"width3")
    print("width3 final",exact_accuracy(model,300000,0),exact_accuracy(model,300000,.7),flush=True)
    export(model)
    model = prune(model,2)
    train_stage(model,30000,1.2e-5,0.45,"width2a")
    train_stage(model,16000,4e-6,0.40,"width2b")
    print("width2 final",exact_accuracy(model,1000000,0),exact_accuracy(model,1000000,.7),flush=True)
    torch.save(model.state_dict(),"/workspace/final.pt")
    export(model)

if __name__ == "__main__": main()
