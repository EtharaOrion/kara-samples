import time, torch, data, json
import torch.nn.functional as F
from torch.func import functional_call, vmap
from model_src import Adder, BASE

CFG = '{"d":3,"code_dim":2,"f1":3,"f1_in":[0,2],"f1_out":[2,3],"qk_in":[0,3],"v_in":[0,3],"o_out":[2,3],"f2":3,"f2_in":[0,3],"f2_out":[0,2],"r_in":[0,2],"read_id":true,"r_bias":false,"norm":""}'
cfg = dict(BASE); cfg.update(json.loads(CFG))
base = Adder(cfg).cuda()
f = lambda pp, xx: functional_call(base, pp, (xx,))
g = torch.Generator(device="cuda").manual_seed(0)

for E, B in [(256, 1024), (256, 4096), (256, 16384), (1024, 4096)]:
    p = {k: torch.randn((E,) + tuple(v.shape), device="cuda", requires_grad=True)
         for k, v in base.named_parameters()}
    dig = data.train_batch(B, "cuda", g)
    x, y = data.tokens(dig), data.targets(dig)
    yr = y.repeat(E, 1).reshape(-1)

    def step(gen):
        if gen:
            d2 = data.train_batch(B, "cuda", g)
            xx, yy = data.tokens(d2), data.targets(d2)
            yy = yy.repeat(E, 1).reshape(-1)
        else:
            xx, yy = x, yr
        lg = vmap(f, in_dims=(0, None))(p, xx)[:, :, 1:, :]
        l = F.cross_entropy(lg.reshape(-1, 10), yy)
        torch.autograd.grad(l, list(p.values()))

    for gen in (False, True):
        for _ in range(5):
            step(gen)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(30):
            step(gen)
        torch.cuda.synchronize(); dt = (time.time() - t) / 30
        print(f"E={E} B={B} datagen={int(gen)}: {dt*1000:6.1f} ms/step  "
              f"{1/dt:6.1f} steps/s  {E*B/dt/1e6:6.2f} M member-samples/s", flush=True)
