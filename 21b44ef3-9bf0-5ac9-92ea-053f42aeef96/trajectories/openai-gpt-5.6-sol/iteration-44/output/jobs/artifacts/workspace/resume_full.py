import sys
sys.path.insert(0,'/workspace')
from train_fallback import *
def main():
 torch.manual_seed(777); random.seed(777); torch.backends.cuda.matmul.allow_tf32=True
 m=FullTransformer().cuda(); m.load_state_dict(torch.load('/workspace/full.pt',weights_only=True)['state'])
 opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=.001,fused=True)
 train_phase(m,opt,14000,2e-4,.20,'resume','/workspace/full_resume.pt')
 train_phase(m,opt,8000,3e-5,.35,'polish','/workspace/full_final.pt')
 print('large',evaluate(m,500000,0),evaluate(m,500000,.8),flush=True); torch.save({'state':m.state_dict()},'/workspace/full_final.pt')
if __name__=='__main__': main()
