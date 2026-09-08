import torch
from train import TrainModel, evaluate, export, prune, train_steps


def main():
    model = TrainModel(2).cuda()
    model.load_state_dict(torch.load('/workspace/width2.pt', map_location='cuda', weights_only=True))
    model = prune(model, 1)
    best = 0.0
    for stage, (steps, lr, fraction) in enumerate([(20000, 5e-6, .45), (20000, 2e-6, .55), (20000, 1e-6, .4)]):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .98), weight_decay=0)
        train_steps(model, optimizer, steps, 8192, fraction)
        accuracy, margin = evaluate(model, 200000, .5)
        print('stage', stage, accuracy, margin, flush=True)
        torch.save(model.state_dict(), '/workspace/width1.pt')
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), '/workspace/width1_best.pt')
    if best >= .9995:
        model.load_state_dict(torch.load('/workspace/width1_best.pt', weights_only=True))
        export(model)


if __name__ == '__main__':
    main()
