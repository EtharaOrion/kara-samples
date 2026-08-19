import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace")
from train import AdditionTransformer, autoregressive_errors, batch_data, export_submission


def main():
    torch.manual_seed(3349)
    device = torch.device("cuda")
    model = AdditionTransformer().to(device)
    model.load_state_dict(torch.load("/workspace/final.pt", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, fused=True)
    best_score = 10**9
    best = None
    for step in range(1, 20001):
        aa, bb, prefix, target = batch_data(4096, device, 0.28, 0.10)
        logits = model(aa, bb, prefix)[:, 15:]
        # Carry mistakes later in a sequence spoil exact accuracy; weight all columns equally,
        # while adding a small margin penalty to ambiguous digit decisions.
        flat = logits.reshape(-1, 10)
        labels = target.reshape(-1)
        loss = F.cross_entropy(flat, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if step % 2000 == 0:
            model.eval()
            er, nr = autoregressive_errors(model, 32, 4096, lambda n: batch_data(n, device, 0.0))
            es, ns = autoregressive_errors(model, 16, 4096, lambda n: batch_data(n, device, 0.50, 0.16))
            score = er * 2 + es
            print(f"step {step} loss {loss.item():.8f} random {er}/{nr} mixed {es}/{ns} score {score}", flush=True)
            if score < best_score:
                best_score = score
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best, "/workspace/refined.pt")
            model.train()
    model.load_state_dict(best)
    model.eval()
    export_submission(model)
    torch.save(model.state_dict(), "/workspace/final.pt")
    print("exported", best_score, flush=True)

if __name__ == "__main__": main()
