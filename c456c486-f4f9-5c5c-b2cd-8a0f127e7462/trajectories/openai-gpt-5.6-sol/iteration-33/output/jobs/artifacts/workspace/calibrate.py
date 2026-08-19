import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import AdditionTransformer, autoregressive_errors, batch_data, export_submission


def main():
    torch.manual_seed(3317)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = AdditionTransformer().to(device)
    model.load_state_dict(torch.load("/workspace/final.pt", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001, fused=True)
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = 10**9
    for step in range(1, 16001):
        lr = 5e-5 if step <= 8000 else 2e-5
        optimizer.param_groups[0]["lr"] = lr
        aa, bb, prefix, target = batch_data(4096, device, 0.38, 0.16)
        logits = model(aa, bb, prefix)[:, 15:]
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0:
            model.eval()
            er, nr = autoregressive_errors(model, 8, 4096, lambda n: batch_data(n, device, 0.0))
            es, ns = autoregressive_errors(model, 8, 4096, lambda n: batch_data(n, device, 0.55, 0.20))
            score = er * 4 + es
            print(f"step {step} loss {loss.item():.7f} random {er}/{nr} mixed {es}/{ns} score {score}", flush=True)
            if score < best_score:
                best_score = score
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best, "/workspace/calibrated.pt")
            model.train()
    model.load_state_dict(best)
    model.eval()
    export_submission(model)
    torch.save(model.state_dict(), "/workspace/final.pt")
    print("exported calibrated best", best_score, flush=True)


if __name__ == "__main__":
    main()
