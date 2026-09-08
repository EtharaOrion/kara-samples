"""Rank saved members by held-out exact match across widths, and report structure."""
import sys, torch, arch, data, train2 as T
dev="cuda"; ck=sys.argv[1]
st=torch.load(ck,map_location=dev); P={k:v.to(dev) for k,v in st["p"].items()}; cfg=st["cfg"]
E=P["code"].shape[0]; g=torch.Generator(device=dev).manual_seed(4242)
tot=torch.zeros(E,device=dev)
for n in (2,3,5,8,11,15):
    a,_=T.evaluate(P,cfg,n,dev,g,N=8192,chunk=1024); tot+=a
    print(f"  n={n:2d}: #==1.0 {int((a>=1.0).sum()):3d}  #>=.99 {int((a>=0.99).sum()):3d}  max {a.max():.5f}")
tot/=6
order=torch.argsort(tot,descending=True)
print("top 12 mean-acc:", [round(float(tot[i]),5) for i in order[:12]])
# structure: does the bank separate absorb/transparent/generate on s = a+b?
A=torch.arange(10,device=dev)
for r,i in enumerate(order[:8]):
    p={k:v[i:i+1] for k,v in P.items()}
    code=arch.full_code(p,cfg)[0,:,0].double()
    d=torch.arange(10.,device=dev,dtype=torch.float64); sig=float((code*d).sum()/(d*d).sum())
    x=code[A][:,None]+code[A][None,:]
    z=x[...,None]*p["Bw"][0,0].double()+p["bb"][0].double(); u=z.clamp(0,1)
    k=(u*p["kw"][0].double()).sum(-1); v=(u*p["vw"][0].double()).sum(-1)+p["vb"][0].double()
    s=(A[:,None]+A[None,:])
    sat=float(torch.where(z<=0,-z,z-1).min())
    kc=[ (float(k[s<=8].min()),float(k[s<=8].max())), (float(k[s==9].min()),float(k[s==9].max())),
         (float(k[s>=10].min()),float(k[s>=10].max())) ]
    vc=[ (float(v[s<=8].min()),float(v[s<=8].max())), (float(v[s==9].min()),float(v[s==9].max())),
         (float(v[s>=10].min()),float(v[s>=10].max())) ]
    knees=sorted([float(-p['bb'][0,j]/p['Bw'][0,0,j])/sig for j in range(cfg['U'])])
    print(f"\n#{r} idx {int(i)} acc {float(tot[i]):.5f} sigma {sig:+.4f} knees/sigma {[round(q,3) for q in knees]} "
          f"sat_slack {sat:+.4f}\n   w1/s {float(p['w1'][0,0])/sig:+.3f} w2/s {float(p['w2'][0,0])/sig:+.3f} "
          f"lam {float(p['lam'][0]):+.3f} q {float(p['q'][0]):+.3f} kw {[round(float(t),3) for t in p['kw'][0]]}"
          f"\n   key  abs {kc[0]} trans {kc[1]} gen {kc[2]}\n   val  abs {vc[0]} trans {vc[1]} gen {vc[2]}")
