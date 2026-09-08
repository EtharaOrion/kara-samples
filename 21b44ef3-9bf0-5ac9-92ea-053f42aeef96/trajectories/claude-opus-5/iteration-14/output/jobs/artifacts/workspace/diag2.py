import torch, arch, data, train as T
dev="cuda"
st=torch.load("ckpt/phase1.pt",map_location=dev); p={k:v.to(dev) for k,v in st["p"].items()}; cfg=st["cfg"]
E=p["code"].shape[0]; print("cfg",cfg,"members",E)
code=arch.full_code(p,cfg)[:,:,0]
print("member0 code   ", [round(float(x),3) for x in code[0]])
for k in ("Bw","bb","kw","vw","vb","q","lam","w1","w2","ls","rb"):
    print(f"  {k:4s}", [round(float(x),4) for x in p[k][0].flatten()])
# knee of the single unit, in units of the code step
step=float(code[0][1]-code[0][0])
print("  knee/step =", round(float(-p['bb'][0,0]/p['Bw'][0,0,0])/step,3), " step",round(step,4))

# is the routing right when nothing is transparent?
def acc_regime(p,cfg,n,regime,N=8192):
    g=torch.Generator(device=dev).manual_seed(1)
    da,db=data.sample_raw(N,n,regime,dev,g)
    if regime=="notrans":  pass
    ta,tb=data.to_tokens(da,db); tgt=data.targets(da,db)
    pred=arch.forward(p,ta,tb,cfg).argmax(-1)
    return (pred[:,:,1:]==tgt[None,:,1:]).all(-1).float().mean(1)

# custom: no transparent place at all (absorb/generate only)
def acc_notrans(p,cfg,n,N=8192):
    g=torch.Generator(device=dev).manual_seed(2)
    r=torch.rand(N,n,device=dev,generator=g)
    cls=torch.where(r<0.5,0,2)
    da,db=data._digits_for_class(cls,N,n,dev,g)
    ta,tb=data.to_tokens(da,db); tgt=data.targets(da,db)
    pred=arch.forward(p,ta,tb,cfg).argmax(-1)
    return (pred[:,:,1:]==tgt[None,:,1:]).all(-1).float().mean(1)

for n in (1,2,3,5,8):
    a=acc_notrans(p,cfg,n); b=acc_regime(p,cfg,n,"uniform")
    print(f"n={n}: no-transparent acc max {a.max():.4f} mean {a.mean():.4f} | uniform acc max {b.max():.4f}")
