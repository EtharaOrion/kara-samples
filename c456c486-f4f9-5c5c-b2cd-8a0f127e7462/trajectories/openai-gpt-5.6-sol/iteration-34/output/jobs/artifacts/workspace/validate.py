import torch
import train


def decode(model, a, b):
    n = a.shape[0]
    ad = train.digits(a, 14)
    bd = train.digits(b, 14)
    ad = torch.cat((ad, torch.full((n, 1), 10, device="cuda", dtype=torch.long)), 1)
    bd = torch.cat((bd, torch.full((n, 1), 10, device="cuda", dtype=torch.long)), 1)
    previous = torch.full((n, 1), 10, device="cuda", dtype=torch.long)
    emitted = []
    for _ in range(15):
        digit = model(ad, bd, previous).argmax(1)
        emitted.append(digit)
        previous = torch.cat((previous, digit[:, None]), 1)
    return (torch.stack(emitted, 1) * (10 ** torch.arange(15, device="cuda"))).sum(1)


def systematic():
    cap = 10 ** 14 - 1
    pairs = {(0, 0), (cap, 0), (cap, cap), (cap, 1), (1, cap)}
    for p in range(14):
        q = 10 ** p
        for x in range(10):
            for y in range(10):
                pairs.add((x * q, y * q))
        for length in range(1, 15 - p):
            run = (10 ** length - 1) * q
            pairs.add((run, q))
            pairs.add((q, run))
            pairs.add((run, 0))
            if run <= cap:
                pairs.add((run, cap - run))
    for length in range(1, 15):
        run = 10 ** length - 1
        pairs.add((run, 1))
        pairs.add((run, run))
        pairs.add((run, 0))
    pairs = [(a,b) for a,b in pairs if 0 <= a <= cap and 0 <= b <= cap]
    return pairs


def main():
    torch.manual_seed(3401)
    m = train.Adder().cuda()
    m.load_state_dict(torch.load("/workspace/final.pt", weights_only=True))
    m.eval()
    print("params", sum(p.numel() for p in m.parameters()))
    with torch.no_grad():
        for structured in (False, True):
            good = total = 0
            for _ in range(64):
                a,b,_,target,av,bv=train.make_batch(8192,structured)
                prediction=decode(m,av,bv)
                expected=(target * (10 ** torch.arange(15,device="cuda"))).sum(1)
                good += (prediction == expected).sum().item(); total += len(av)
            print("structured" if structured else "uniform",good,total,total-good)
        pairs=systematic(); good=0; failures=[]
        for start in range(0,len(pairs),8192):
            part=pairs[start:start+8192]
            av=torch.tensor([x[0] for x in part],device="cuda",dtype=torch.long)
            bv=torch.tensor([x[1] for x in part],device="cuda",dtype=torch.long)
            out=decode(m,av,bv).cpu().tolist()
            for pair,pred in zip(part,out):
                if pred == pair[0]+pair[1]: good += 1
                elif len(failures)<20: failures.append((pair,pred,pair[0]+pair[1]))
        print("systematic",good,len(pairs),len(pairs)-good,failures)

        # Compare first-block attention scores for two input sequences.
        a1,b1,p1,_,_,_=train.make_batch(1,False)
        a2,b2,p2,_,_,_=train.make_batch(1,False)
        block=m.blocks[0]
        def scores(a,b,p):
            x=torch.cat((m.a_embed(a)+m.b_embed(b),m.out_embed(p)),1)+m.position
            y=block.norm1(x); q,k,_=block.qkv(y).chunk(3,-1)
            q=q.view(1,30,3,3).transpose(1,2); k=k.view(1,30,3,3).transpose(1,2)
            return q @ k.transpose(-2,-1)
        delta=(scores(a1,b1,p1)-scores(a2,b2,p2)).abs().max().item()
        before=decode(m,*[z for z in (torch.tensor([12345678901234],device="cuda"),torch.tensor([7654321098765],device="cuda"))])
        saved=m.head.weight.detach().clone(); m.head.weight.zero_()
        after=decode(m,torch.tensor([12345678901234],device="cuda"),torch.tensor([7654321098765],device="cuda"))
        m.head.weight.copy_(saved)
        print("attention_delta",delta,"head_zero_changed",before.item(),after.item(),before.item()!=after.item())

if __name__ == "__main__": main()
