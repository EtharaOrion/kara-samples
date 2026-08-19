import random, time
from submission import build_model, add
m,meta=build_model()
r=random.Random(918273)
N=1000
correct=0
start=time.time()
for i in range(N):
    a=r.randrange(100_000_000_000_000); b=r.randrange(100_000_000_000_000)
    got=add(m,a,b)
    correct += got == a+b
    if (i+1)%50==0: print(i+1,correct,correct/(i+1),time.time()-start,flush=True)
