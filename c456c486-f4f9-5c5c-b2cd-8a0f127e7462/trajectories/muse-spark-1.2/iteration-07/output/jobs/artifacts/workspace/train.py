import torch
import torch.nn as nn
import random
import math
from submission import TinyTransformer, VOCAB_SIZE, D_MODEL, NHEAD, D_FF, MAX_LEN, INPUT_LEN, OUTPUT_LEN, PAD_TOKEN, PLUS_TOKEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_VAL = 99_999_999_999_999
BATCH = 512
STEPS = 60000
LR = 4e-4
WD = 0.01

def encode_pair(a,b):
    toks=[]
    for _ in range(14):
        toks.append(a%10); a//=10
    toks.append(PLUS_TOKEN)
    for _ in range(14):
        toks.append(b%10); b//=10
    return toks

def encode_sum(s):
    toks=[]
    for _ in range(15):
        toks.append(s%10); s//=10
    return toks

def sample_batch(bs):
    inp = torch.zeros(bs, INPUT_LEN+OUTPUT_LEN, dtype=torch.long)
    tgt = torch.zeros(bs, INPUT_LEN+OUTPUT_LEN, dtype=torch.long)
    # we will create full sequence: input(29) + output(15) =44
    # input part is encode_pair, output part is encode_sum
    # For training, we do teacher forcing on full sequence, loss only on output positions
    for i in range(bs):
        r = random.random()
        if r < 0.35:
            a = random.randint(0, MAX_VAL)
            b = random.randint(0, MAX_VAL)
        elif r < 0.50:
            # digit-length uniform
            da = random.randint(1,14)
            db = random.randint(1,14)
            a = random.randint(10**(da-1) if da>1 else 0, 10**da -1)
            b = random.randint(10**(db-1) if db>1 else 0, 10**db -1)
            if da==1 and random.random()<0.5:
                a = random.randint(0,9)
            if db==1 and random.random()<0.5:
                b = random.randint(0,9)
        elif r < 0.62:
            # carry-heavy digits 5-9
            a = 0; b=0
            for pos in range(14):
                da = random.randint(5,9)
                db = random.randint(5,9)
                a += da * (10**pos)
                b += db * (10**pos)
            # randomize a bit: sometimes mix
            if random.random()<0.5:
                a = random.randint(0, MAX_VAL)
            if random.random()<0.5:
                b = random.randint(0, MAX_VAL)
        elif r < 0.72:
            # zeros
            if random.random()<0.5:
                a = 0
                b = random.randint(0, MAX_VAL)
            else:
                a = random.randint(0, MAX_VAL)
                b = 0
            if random.random()<0.3:
                a=0; b=0
        elif r < 0.82:
            # powers of 10 and 5*10^k for long carries
            k = random.randint(0,13)
            base = 10**k
            choices = [base, 5*base, 10**14 -1, 10**k -1]
            a = random.choice(choices) % (MAX_VAL+1)
            b = random.choice(choices) % (MAX_VAL+1)
            # also include 999... patterns
            if random.random()<0.3:
                a = int("9"*random.randint(1,14)) if random.random()<0.5 else a
                b = int("9"*random.randint(1,14)) if random.random()<0.5 else b
        elif r < 0.90:
            # trailing 9s
            n9a = random.randint(1,7)
            n9b = random.randint(1,7)
            a = random.randint(0, MAX_VAL // (10**n9a)) * (10**n9a) + int("9"*n9a)
            b = random.randint(0, MAX_VAL // (10**n9b)) * (10**n9b) + int("9"*n9b)
            a = min(a, MAX_VAL); b=min(b, MAX_VAL)
        else:
            # small numbers
            a = random.randint(0, 10000)
            b = random.randint(0, 10000)

        s = a + b
        inp_tokens = encode_pair(a,b)
        out_tokens = encode_sum(s)
        full = inp_tokens + out_tokens
        inp[i] = torch.tensor(full, dtype=torch.long)
        tgt[i] = torch.tensor(full, dtype=torch.long)
    return inp, tgt

def main():
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    ce = nn.CrossEntropyLoss()

    best_acc = 0
    for step in range(1, STEPS+1):
        model.train()
        x, y = sample_batch(BATCH)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # input to model is x[:, :-1], target is y[:, 1:] but we have full 44 length
        # Actually we want to predict next token: logits for positions 0..42 predict token 1..43
        # Loss only on output part: positions INPUT_LEN .. 43 (since output tokens are at 29..43)
        # Input: x[:, :-1] (43 tokens), target: y[:, 1:] (43 tokens)
        logits = model(x[:, :-1])  # (B,43,V)
        targets = y[:, 1:]  # (B,43)
        # mask: only positions where target corresponds to output tokens
        # x[:, :-1] positions 0..42, targets 1..43
        # output tokens are at indices 29..43 in full sequence
        # So targets positions 28..42 correspond to output tokens (since target index = input index+1)
        # Let's compute mask: target position j in 0..42 corresponds to full index j+1
        # We want full index >= INPUT_LEN (29) => j+1 >=29 => j>=28
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, 28:] = True
        logits_m = logits[mask]  # (B*15, V)
        targets_m = targets[mask]  # (B*15,)
        loss = ce(logits_m, targets_m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0:
            # quick eval
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 1000
                for _ in range(total//BATCH +1):
                    bs = min(BATCH, total - correct - (total//BATCH)*0)  # just do 1000
                    pass
                # simpler: sample 1000 random pairs and test via model forward (teacher forcing accuracy)
                # do token-level accuracy on output
                x_e, y_e = sample_batch(512)
                x_e = x_e.to(DEVICE); y_e=y_e.to(DEVICE)
                logits_e = model(x_e[:,:-1])
                preds = logits_e.argmax(dim=-1)
                targets_e = y_e[:,1:]
                mask_e = torch.zeros_like(targets_e, dtype=torch.bool)
                mask_e[:,28:]=True
                acc = (preds[mask_e]==targets_e[mask_e]).float().mean().item()
                # sequence accuracy: all 15 output tokens correct
                preds_seq = preds[mask_e].view(-1,15)
                targets_seq = targets_e[mask_e].view(-1,15)
                seq_acc = (preds_seq==targets_seq).all(dim=1).float().mean().item()
            print(f"step {step} loss {loss.item():.4f} tok_acc {acc:.4f} seq_acc {seq_acc:.4f} lr {sched.get_last_lr()[0]:.6f}")
            if seq_acc > best_acc:
                best_acc = seq_acc

        if step % 5000 == 0 or step==STEPS:
            # full add() eval on 2000 random pairs
            model.eval()
            with torch.no_grad():
                from submission import add
                correct=0
                for _ in range(2000):
                    a = random.randint(0, MAX_VAL)
                    b = random.randint(0, MAX_VAL)
                    pred = add(model, a,b)
                    if pred == a+b:
                        correct+=1
                print(f"  >>> add() eval 2000 random: {correct}/2000 = {correct/2000:.4f}")
                # edge cases
                edges = [(0,0),(0,MAX_VAL),(MAX_VAL,0),(MAX_VAL,MAX_VAL),(MAX_VAL,1),(12345678901234,98765432109876),(50000000000000,50000000000000)]
                for a,b in edges:
                    pred = add(model,a,b)
                    ok = "OK" if pred==a+b else f"FAIL got {pred} exp {a+b}"
                    print(f"    edge {a}+{b} -> {ok}")

    # final save: write weights into submission.py
    print("Training done, saving weights to submission.py")
    # Read current submission.py
    with open("/workspace/submission.py","r") as f:
        content = f.read()
    # Prepare weight injection code
    # We'll append code that loads weights, but better to embed as base64? No, forbidden.
    # Instead we will save state_dict and generate a new submission.py that loads it via torch tensors embedded as python literals
    # The simplest: create a new submission.py with weights hardcoded as lists, and load them in build_model
    # But we need to keep submission.py clean: only torch imports and model definition + weight loading
    # Approach: serialize state_dict to a python file that sets weights via torch.tensor literals
    # We'll generate a new submission.py that includes the model class and then sets state_dict from embedded tensors

    # Get state dict
    sd = model.state_dict()
    # Generate python code for weights
    import textwrap
    # Create new submission content with embedded weights
    # We'll keep the same class definition but add a function to load weights
    # Instead, we will create a new file that directly sets the parameters via torch.tensor

    # Build weight initialization code
    weight_code_lines = []
    weight_code_lines.append("def _load_weights(model):")
    weight_code_lines.append("    import torch")
    weight_code_lines.append("    sd = {}")
    for k, v in sd.items():
        # convert to list
        lst = v.cpu().numpy().tolist()
        # Use repr for compactness
        weight_code_lines.append(f"    sd['{k}'] = torch.tensor({repr(lst)}, dtype=torch.float32)")
        # For integer? all float32, but embedding weight is float
    weight_code_lines.append("    model.load_state_dict(sd)")
    weight_code_lines.append("    return model")
    weight_code = "\n".join(weight_code_lines)

    # Now build new submission.py
    # Read original to get class definition up to build_model
    # We'll construct fresh
    new_sub = open("/workspace/submission.py","r").read()
    # Append weight loading: modify build_model to call _load_weights
    # Find build_model definition and replace
    old_build = """def build_model():
    model = TinyTransformer()
    metadata = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "d_ff": D_FF,
        "n_layers": N_LAYERS,
        "max_len": MAX_LEN,
    }
    return model, metadata"""

    new_build = weight_code + "\n\ndef build_model():\n    model = TinyTransformer()\n    _load_weights(model)\n    metadata = {\n        \"vocab_size\": VOCAB_SIZE,\n        \"d_model\": D_MODEL,\n        \"nhead\": NHEAD,\n        \"d_ff\": D_FF,\n        \"n_layers\": N_LAYERS,\n        \"max_len\": MAX_LEN,\n    }\n    return model, metadata"

    if old_build in new_sub:
        new_sub = new_sub.replace(old_build, new_build)
    else:
        # fallback: append
        new_sub += "\n\n" + weight_code + "\n"

    with open("/workspace/submission.py","w") as f:
        f.write(new_sub)
    print("Saved new submission.py with embedded weights")
    # Verify
    import importlib, sys
    if 'submission' in sys.modules:
        del sys.modules['submission']
    import submission as sub2
    import importlib as imp
    imp.reload(sub2)
    m2, _ = sub2.build_model()
    m2.eval()
    with torch.no_grad():
        for a,b in [(0,0),(0,MAX_VAL),(MAX_VAL,MAX_VAL),(12345678901234,98765432109876)]:
            pred = sub2.add(m2,a,b)
            print(f"verify {a}+{b}={a+b} pred={pred} {'OK' if pred==a+b else 'FAIL'}")
        # random 500
        ok=0
        for _ in range(500):
            a=random.randint(0,MAX_VAL); b=random.randint(0,MAX_VAL)
            if sub2.add(m2,a,b)==a+b:
                ok+=1
        print(f"verify random 500: {ok}/500 {ok/500:.4f}")
    # param count
    total_params = sum(p.numel() for p in m2.parameters())
    print(f"Total params: {total_params}")

if __name__ == "__main__":
    main()
