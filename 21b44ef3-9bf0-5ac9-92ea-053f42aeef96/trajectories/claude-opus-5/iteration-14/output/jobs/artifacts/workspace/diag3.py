import sys, torch, arch, data
dev="cuda"; ck=sys.argv[1] if len(sys.argv)>1 else "ckpt/c11.pt"
st=torch.load(ck,map_location=dev); P={k:v.to(dev) for k,v in st["p"].items()}; cfg=st["cfg"]
i=0
p={k:v[i:i+1] for k,v in P.items()}
code=arch.full_code(p,cfg)[0,:,0]
d=torch.arange(10.,device=dev); sig=float((code*d).sum()/(d*d).sum())
print("code      ", [round(float(x),4) for x in code])
print("code/sigma", [round(float(x)/sig,3) for x in code], " sigma",round(sig,4))
for k in ("Bw","bb","kw","vw","vb","q","lam","w1","w2","rb","ls"):
    print(f"  {k:3s}", [round(float(x),4) for x in p[k].flatten()])
print("  w1/sigma", round(float(p['w1'][0,0])/sig,3), "  w2/sigma", round(float(p['w2'][0,0])/sig,3))
# bank / key / value as a function of s = a+b
A=torch.arange(10,device=dev)
x=code[A][:,None]+code[A][None,:]
u=(x[...,None]*p["Bw"][0,0]+p["bb"][0]).clamp(0,1)
k=(u*p["kw"][0]).sum(-1); v=(u*p["vw"][0]).sum(-1)+p["vb"][0]
s=(A[:,None]+A[None,:])
print("\n  s : key(min,max)  value(min,max)")
for sv in range(19):
    m=s==sv
    print(f"  {sv:2d}: k [{k[m].min():+.3f},{k[m].max():+.3f}]  v [{v[m].min():+.3f},{v[m].max():+.3f}]")
# attention profile and per-class accuracy at n=8
g=torch.Generator(device=dev).manual_seed(7)
ta,tb,tgt,da,db=data.batch(4096,8,dev,g,heldout=True)
want=[]; dlog=arch.forward(p,ta,tb,cfg,want=want); w=want[0]
pred=dlog.argmax(-1)
hit=(pred[0,:,1:]==tgt[:,1:])
print("\n  per-position digit acc:", [round(float(x),3) for x in hit.float().mean(0)])
sm=da+db; cls=torch.where(sm<=8,0,torch.where(sm==9,1,2))
for c,nm in ((0,"absorb"),(1,"transp"),(2,"gener ")):
    m=cls==c
    print(f"  class {nm} acc {hit[:,:8][m].float().mean():.4f}  n={int(m.sum())}")
a_s=w["a_s"][0].mean(0); print("\n  mean strict-attn row for query 5:", [round(float(x),3) for x in a_s[5]])
a_i=w["a_i"][0].mean(0); print("  mean incl-attn  row for query 5:", [round(float(x),3) for x in a_i[5]])
# what would ideal carries give?
cin_true=torch.zeros_like(sm); carry=torch.zeros(sm.shape[0],device=dev,dtype=torch.long)
for j in range(8):
    cin_true[:,j]=carry; carry=((sm[:,j]+carry)>=10).long()
print("  corr(c_in_model, true cin) =", round(float(torch.corrcoef(torch.stack([w["c_in"][0,:,1:9].flatten(),cin_true.float().flatten()]))[0,1]),4))
