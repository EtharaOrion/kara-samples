import argparse
import json
import time

import torch
from torch import nn

from train import Model, evaluate, examples


def edge_examples(batch, device):
    powers8 = 10 ** torch.arange(8, device=device)
    powers9 = 10 ** torch.arange(9, device=device)
    a = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    b = torch.randint(10_000_000, 100_000_000, (batch,), device=device)
    quarter = batch // 4
    # Near both numeric boundaries.
    a[:quarter] = torch.randint(10_000_000, 10_010_000, (quarter,), device=device)
    b[:quarter] = torch.randint(99_990_000, 100_000_000, (quarter,), device=device)
    # Dense high digits stress carry propagation into the ninth result digit.
    high = torch.randint(8, 10, (quarter, 8), device=device)
    high[:, 7] = 9
    a[quarter:2*quarter] = (high * powers8).sum(1)
    high = torch.randint(8, 10, (quarter, 8), device=device)
    high[:, 7] = 9
    b[quarter:2*quarter] = (high * powers8).sum(1)
    # Long runs of trailing nines with randomized prefixes.
    run = torch.randint(1, 8, (quarter,), device=device)
    base = 10 ** run
    a[2*quarter:3*quarter] = (torch.randint(10_000_000, 100_000_000, (quarter,), device=device) // base) * base + base - 1
    b[2*quarter:3*quarter] = torch.randint(10_000_000, 100_000_000, (quarter,), device=device)
    # Sparse, repeated, and round-number patterns are rare under uniform sampling.
    n = batch - 3 * quarter
    patterns = torch.tensor([
        10_000_000, 10_000_001, 10_000_009, 10_000_099, 10_000_999,
        10_009_999, 10_099_999, 10_999_999, 50_000_000, 88_888_888,
        89_999_999, 90_000_000, 98_888_888, 99_000_000, 99_900_000,
        99_990_000, 99_999_000, 99_999_900, 99_999_990, 99_999_998,
        99_999_999], device=device)
    a[3*quarter:] = patterns[torch.randint(len(patterns), (n,), device=device)]
    b[3*quarter:] = patterns[torch.randint(len(patterns), (n,), device=device)]
    ad = (a[:, None] // powers8) % 10
    bd = (b[:, None] // powers8) % 10
    result = ((a + b)[:, None] // powers9) % 10
    operands = torch.stack((ad, bd), dim=2).reshape(batch, 16)
    context = torch.cat((operands, torch.full((batch, 1), 10, device=device)), 1)
    return torch.cat((context, result[:, :-1]), 1), result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=4096)
    args = p.parse_args()
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    c = ckpt['config']
    model = Model(c['width'], c['heads'], c['ff']).cuda()
    model.load_state_dict(ckpt['state'])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
    loss_fn = nn.CrossEntropyLoss()
    started = time.time()
    for step in range(1, args.steps + 1):
        if step % 2:
            inputs, targets = edge_examples(args.batch, model.token.weight.device)
        else:
            inputs, targets, _, _ = examples(args.batch, model.token.weight.device)
        logits = model(inputs)[:, 16:25]
        loss = loss_fn(logits.reshape(-1, 10), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            exact, digit = evaluate(model, batches=2)
            print(json.dumps({'step':step,'loss':loss.item(),'exact':exact,'digit':digit,
                              'seconds':round(time.time()-started,1)}), flush=True)
    exact, digit = evaluate(model, batches=16)
    ckpt.update(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                exact=exact, digit=digit, edge_finetuned=True)
    torch.save(ckpt, args.output)
    print(json.dumps({'exact':exact,'digit':digit,'parameters':ckpt['parameters']}))

if __name__ == '__main__':
    main()
