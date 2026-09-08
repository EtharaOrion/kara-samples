"""Pick the best member of a checkpoint by held-out exact match across widths, emit it."""
import sys, torch, arch, data, build, train2 as T
import os
dev=os.environ.get("DEV","cuda"); ck=sys.argv[1]; out=sys.argv[2]; params=sys.argv[3] if len(sys.argv)>3 else ""
st=torch.load(ck,map_location=dev); P={k:v.to(dev) for k,v in st["p"].items()}; cfg=st["cfg"]
E=P["code"].shape[0]; g=torch.Generator(device=dev).manual_seed(31337)
tot=torch.zeros(E,device=dev)
for n in (2,3,5,8,11,15):
    a,_=T.evaluate(P,cfg,n,dev,g,N=int(os.environ.get("EVN",16384)),chunk=1024); tot+=a
tot/=6
i=int(torch.argmax(tot)); print(f"best member {i}: mean held-out exact {float(tot[i]):.6f}")
p={k:v[i].cpu() for k,v in P.items()}
keys=params.split(",") if params else [k for k in build.ALL_KEYS if k in p]
n,sz=build.emit(p,cfg,keys,out,sys.argv[4] if len(sys.argv)>4 else "")
print(f"wrote {out}: {n} parameters, {sz} bytes")
