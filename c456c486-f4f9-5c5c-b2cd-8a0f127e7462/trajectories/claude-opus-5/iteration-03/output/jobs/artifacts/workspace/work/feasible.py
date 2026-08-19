"""Feasibility check ONLY: does the d_model/d_ff/head architecture *admit* an exact
carry-lookahead solution?  Constructs weights in closed form and measures accuracy.
Nothing here is exported -- the shipped model's weights come from training."""
import torch, tiny

dev = "cuda"
cfg = tiny.Cfg(d_model=3, d_ff=4, head="linear")
tab = tiny.pair_row_table(dev)
rel, causal = tiny.make_masks(dev)
z = lambda *s: torch.zeros(1, *s, device=dev)
p = {k: z(*v.shape[1:]) for k, v in tiny.init_params(cfg, 1, dev, 0).items()}

LAM, SIG, D = 1.0 / 9.5, 12.0, 400.0
p["U"][0] = torch.arange(10, device=dev).float()          # ch0 = a+b = s
prop = (torch.arange(10, device=dev)[:, None] + torch.arange(10, device=dev)[None, :]) == 9
P = torch.zeros(55, device=dev); P[tab[prop]] = -1.0      # ch1 = -1 iff propagate
p["P"][0] = P
p["bq"][0], p["wk"][0, 1], p["slope"][0] = 1.0, D, SIG    # score = D*P_j + slope*(j-i)
p["wv"][0, 0], p["wo"][0, 2] = 1.0, 1.0                   # attend->value = s_m, into ch2

# 4 ReLUs: carry = [s_m>=10], wrap = [s_i + carry >= 10]
p["W1"][0] = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0], [LAM, LAM, LAM, LAM]], device=dev)
p["b1"][0] = torch.tensor([-9 * LAM, -10 * LAM, -9.96, -9.99], device=dev)
p["W2"][0, :, 0] = torch.tensor([1 / LAM, -1 / LAM, -10 / 0.03, 10 / 0.03], device=dev)
k = torch.arange(10, device=dev).float()                  # nearest-centre readout on ch0
p["Wr"][0, 0] = 4.0 * k
p["br"][0] = -2.0 * k ** 2

g = torch.Generator(device=dev).manual_seed(11)
with torch.no_grad():
    for nm, rp in [("uniform", (1., 0, 0, 0)), ("stress", (0, .15, .25, .6))]:
        ok = n = 0
        for _ in range(20):
            a, b, t = tiny.sample_batch(50000, dev, g, regime_p=rp, ndig=14)
            ok += (tiny.forward(p, cfg, a, b, tab, rel, causal).argmax(-1)[0] == t).all(1).sum().item()
            n += 50000
        print(f"constructed {nm}: {ok}/{n} = {ok/n:.6f}")
