import os
os.environ.pop("PYTHONPATH", None)
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

D = 16
H = 2
W = 5

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.n2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, 2 * D)
        self.fc2 = nn.Linear(2 * D, D)

    def forward(self, x):
        z = self.n1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        n, t, _ = q.shape
        q = q.view(n, t, H, D // H).transpose(1, 2)
        k = k.view(n, t, H, D // H).transpose(1, 2)
        v = v.view(n, t, H, D // H).transpose(1, 2)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        mask &= torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1 - W)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
        s = s.masked_fill(~mask, -torch.inf)
        x = x + self.proj((s.softmax(-1) @ v).transpose(1, 2).reshape(n, t, D))
        z = self.n2(x)
        return x + self.fc2(F.gelu(self.fc1(z)))

class Model(nn.Module):
    def __init__(self, sharing="none"):
        super().__init__()
        self.digit = nn.Embedding(10, D)
        self.role = nn.Embedding(3, D)
        self.b0 = Block()
        self.b1 = self.b0 if sharing == "block" else Block()
        if sharing == "mlp":
            self.b1.n2 = self.b0.n2
            self.b1.fc1 = self.b0.fc1
            self.b1.fc2 = self.b0.fc2
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10)

    def forward(self, x):
        roles = torch.arange(x.shape[1], device=x.device).remainder(3)
        x = self.digit(x) + self.role(roles)
        x = self.b0(x)
        x = self.b1(x)
        return self.head(self.norm(x))

def batch(n, device, structured=False):
    lim = 100_000_000_000_000
    a = torch.randint(lim, (n,), device=device)
    b = torch.randint(lim, (n,), device=device)
    if structured:
        # Full-pair patterns enrich long carry chains without enumerating local states.
        pick = torch.rand(n, device=device) < .35
        repeated = torch.randint(10, (n,), device=device)
        powers = torch.tensor([10 ** i for i in range(15)], device=device)
        repnum = (repeated[:, None] * powers).sum(1)
        a = torch.where(pick, repnum.remainder(lim), a)
    c = a + b
    p = torch.tensor([10 ** i for i in range(15)], device=device)
    ad = (a[:, None] // p).remainder(10)
    bd = (b[:, None] // p).remainder(10)
    cd = (c[:, None] // p).remainder(10)
    x = torch.stack((ad, bd, cd), 2).reshape(n, 45)
    return x

@torch.inference_mode()
def greedy(model, a, b):
    aa = list(map(int, reversed(f"{a:015d}")))
    bb = list(map(int, reversed(f"{b:015d}")))
    seq = []
    out = []
    for i in range(15):
        seq.extend((aa[i], bb[i]))
        z = torch.tensor(seq, device=next(model.parameters()).device).unsqueeze(0)
        d = int(model(z)[0, -1].argmax())
        seq.append(d)
        out.append(d)
    return int("".join(map(str, reversed(out))))

@torch.inference_mode()
def evaluate(model, n=1000, seed=9182):
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        a = rng.randrange(100_000_000_000_000)
        b = rng.randrange(100_000_000_000_000)
        ok += greedy(model, a, b) == a + b
    return ok / n

def export(model, sharing, path="/workspace/submission.py"):
    # named_parameters removes aliases, matching parameters_to_vector ordering.
    vals = torch.nn.utils.parameters_to_vector(model.parameters()).detach().half().cpu().tolist()
    literal = repr(vals)
    source = f'''import math\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\nD=16\nH=2\nW=5\nSHARING={sharing!r}\n\nclass Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.n1=nn.LayerNorm(D)\n        self.qkv=nn.Linear(D,3*D)\n        self.proj=nn.Linear(D,D)\n        self.n2=nn.LayerNorm(D)\n        self.fc1=nn.Linear(D,2*D)\n        self.fc2=nn.Linear(2*D,D)\n    def forward(self,x):\n        z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1); n,t,_=q.shape\n        q=q.view(n,t,H,D//H).transpose(1,2); k=k.view(n,t,H,D//H).transpose(1,2); v=v.view(n,t,H,D//H).transpose(1,2)\n        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()\n        mask &= torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1-W)\n        s=(q@k.transpose(-2,-1))/math.sqrt(D//H)\n        s=s.masked_fill(~mask,-torch.inf)\n        x=x+self.proj((s.softmax(-1)@v).transpose(1,2).reshape(n,t,D))\n        z=self.n2(x)\n        return x+self.fc2(F.gelu(self.fc1(z)))\n\nclass Adder(nn.Module):\n    def __init__(self):\n        super().__init__(); self.digit=nn.Embedding(10,D); self.role=nn.Embedding(3,D); self.b0=Block(); self.b1=self.b0 if SHARING=="block" else Block()\n        if SHARING=="mlp":\n            self.b1.n2=self.b0.n2; self.b1.fc1=self.b0.fc1; self.b1.fc2=self.b0.fc2\n        self.norm=nn.LayerNorm(D); self.head=nn.Linear(D,10)\n    def forward(self,x):\n        roles=torch.arange(x.shape[1],device=x.device).remainder(3)\n        x=self.digit(x)+self.role(roles); x=self.b0(x); x=self.b1(x)\n        return self.head(self.norm(x))\n\n_WEIGHTS={literal}\n\ndef build_model():\n    model=Adder()\n    with torch.no_grad():\n        torch.nn.utils.vector_to_parameters(torch.tensor(_WEIGHTS),model.parameters())\n    model.eval()\n    return model,{{"architecture":"two-pass local causal transformer","training":"random full 14-digit pairs","weight_sharing":SHARING}}\n\ndef add(model,a:int,b:int)->int:\n    left=list(map(int,reversed(f"{{a:015d}}")))\n    right=list(map(int,reversed(f"{{b:015d}}")))\n    tokens=[]; answer=[]\n    device=next(model.parameters()).device\n    with torch.inference_mode():\n        for place in range(15):\n            tokens.extend((left[place],right[place]))\n            x=torch.tensor(tokens,dtype=torch.long,device=device).unsqueeze(0)\n            digit=int(model(x)[0,-1].argmax().item())\n            tokens.append(digit); answer.append(str(digit))\n    return int("".join(reversed(answer)))\n'''
    with open(path, "w") as f:
        f.write(source)
    print("exported", path, "params", sum(p.numel() for p in model.parameters()), "bytes", len(source))

def main():
    sharing = os.environ.get("SHARING", "none")
    steps = int(os.environ.get("STEPS", "3000"))
    bs = int(os.environ.get("BS", "4096"))
    torch.manual_seed(int(os.environ.get("SEED", "7")))
    torch.set_float32_matmul_precision("high")
    device = os.environ.get("DEVICE", "cuda")
    model = Model(sharing).to(device)
    print("sharing", sharing, "parameters", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.005)
    milestones = {int(steps*.55): 1.5e-3, int(steps*.75): 7e-4, int(steps*.9): 3e-4}
    for step in range(1, steps + 1):
        if step in milestones:
            for g in opt.param_groups: g["lr"] = milestones[step]
        x = batch(bs, device, structured=step > int(steps*.7))
        logits = model(x)
        loss = F.cross_entropy(logits[:, 1::3].reshape(-1, 10), x[:, 2::3].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 250 == 0 or step == steps:
            acc = evaluate(model, 200)
            print(step, float(loss), "greedy", acc, "lr", opt.param_groups[0]["lr"], flush=True)
    model.eval(); print("final1000", evaluate(model, 1000))
    export(model, sharing)
    torch.save(model.state_dict(), f"/workspace/model_{sharing}.pt")

if __name__ == "__main__": main()
