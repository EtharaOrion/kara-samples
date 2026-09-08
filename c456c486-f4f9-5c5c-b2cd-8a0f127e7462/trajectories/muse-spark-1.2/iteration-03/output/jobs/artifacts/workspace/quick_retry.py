import subprocess, sys, re, time, random, os
# Try d=12,2,2,16 with different seeds using train.py
# train.py now supports --d_model etc but seed is random each run
# We'll just run it multiple times

for attempt in range(5):
    print(f"\n{'='*60}\nATTEMPT {attempt+1}/5: d=12,2,2,16\n{'='*60}")
    result = subprocess.run(
        ["/opt/openhands-venv/bin/python", "train.py", "--steps", "40000", "--lr", "0.001", "--d_model", "12", "--nhead", "2", "--n_layers", "2", "--d_ff", "16"],
        cwd="/workspace", capture_output=True, text=True, timeout=1200
    )
    out = result.stdout + result.stderr
    print(out[-4000:])
    # check if saved submission has good accuracy
    # look for "held-out 2000" or "Final eval"
    m = re.search(r"held-out 2000:\s*([\d.]+)", out)
    if m:
        acc = float(m.group(1))
        print(f"  -> held-out acc {acc}")
        if acc >= 0.99:
            print(f"  *** FOUND GOOD MODEL acc={acc} ***")
            # verify submission.py
            import importlib
            try:
                import submission
                importlib.reload(submission)
                mod, meta = submission.build_model()
                print(f"  meta {meta}")
                # quick verify
                import torch
                ok = True
                for a,b in [(0,0),(99999999999999,99999999999999),(12345678901234,98765432109876)]:
                    pred = submission.add(mod,a,b)
                    if pred != a+b:
                        ok=False
                        print(f"  FAIL {a}+{b}={a+b} got {pred}")
                if ok:
                    print("  All edge cases OK - stopping")
                    sys.exit(0)
            except Exception as e:
                print(f"  verify error {e}")
    else:
        print("  no acc found, checking Final eval")
        m2 = re.search(r"Final eval 5000:\s*([\d.]+)", out)
        if m2:
            print(f"  Final eval {m2.group(1)}")

print("No good model found in 5 attempts, trying d=12,2,2,20")
for attempt in range(3):
    print(f"\nATTEMPT {attempt+1}/3: d=12,2,2,20")
    result = subprocess.run(
        ["/opt/openhands-venv/bin/python", "train.py", "--steps", "40000", "--lr", "0.001", "--d_model", "12", "--nhead", "2", "--n_layers", "2", "--d_ff", "20"],
        cwd="/workspace", capture_output=True, text=True, timeout=1200
    )
    out = result.stdout + result.stderr
    print(out[-3000:])
    m = re.search(r"held-out 2000:\s*([\d.]+)", out)
    if m and float(m.group(1)) >= 0.99:
        print("FOUND")
        sys.exit(0)
