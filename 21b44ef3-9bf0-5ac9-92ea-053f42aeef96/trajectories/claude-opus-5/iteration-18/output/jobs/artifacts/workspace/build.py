"""Select the best certified member from a stage checkpoint and write
/workspace/submission.py (model class + plain float literals, torch only)."""
import argparse, json, os, re, sys
import torch
import certify


HEADER = '''"""Minimal transformer that adds two integers exactly.

One block, a one-dimensional residual stream, {NP} learned parameters.  The
weights below came out of the training pipeline in this directory
(train_parent.py -> stage2.py{S3} -> build.py); this file holds
only the model and its inference path.
"""
import torch
import torch.nn as nn

'''


def extract(text, tag):
    m = re.search(r'# --- BEGIN %s ---\n(.*?)# --- END %s ---' % (tag, tag), text, re.S)
    return m.group(1).rstrip() + '\n'


def fmt(vals, per_line=3, indent=8):
    out, cur = [], []
    for v in vals:
        cur.append(repr(float(v)))
        if len(cur) == per_line:
            out.append(' ' * indent + ', '.join(cur) + ',')
            cur = []
    if cur:
        out.append(' ' * indent + ', '.join(cur) + ',')
    return '\n'.join(out)


def emit(code_free, knee, fold, buf, path, meta_extra, s3=False):
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ship_src.py')).read()
    np_ = len(code_free) + len(knee) + 1
    body = HEADER.format(NP=np_, S3=' -> stage3.py' if s3 else '') + '\n'
    body += extract(src, 'SHIPPED CLASS') + '\n\n'
    body += '_CODE_FREE = [\n%s\n]\n\n' % fmt(code_free)
    body += '_KNEE = [\n%s\n]\n\n' % fmt(knee)
    body += '_FOLD = [\n%s\n]\n\n\n' % fmt(fold)
    body += '''def build_model():
    """Return (model, metadata).  The weights below are the trained values."""
    model = DigitPairAdder(n_knee=%d, bank_w=%r, key_w=%r,
                           val_w=%r, lam=%r)
    with torch.no_grad():
        model.code_free.copy_(torch.tensor(_CODE_FREE, dtype=torch.float32))
        model.knee.copy_(torch.tensor(_KNEE, dtype=torch.float32))
        model.fold.copy_(torch.tensor(_FOLD, dtype=torch.float32))
    model.eval()
    meta = {
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'parameters': {n: tuple(p.shape) for n, p in model.named_parameters()},
        'architecture': 'one transformer block, 1-D residual stream, 2 heads '
                        '(strictly-causal carry-in, inclusively-causal carry-out) '
                        'sharing one content-dependent key/value stream',
        'tokenisation': 'one token per decimal place, LSB first, (0,0) pad at '
                        'each end; token p emits answer digit p-1',
        'range': 'exact for both operands in [10000000, 99999999]',
%s    }
    return model, meta


''' % (len(knee), tuple(buf['bank_w']), tuple(buf['key_w']), tuple(buf['val_w']),
       float(buf['lam']), meta_extra)
    body += extract(src, 'SHIPPED HELPERS')
    with open(path, 'w') as f:
        f.write(body)
    return np_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--out', type=str, default='/workspace/submission.py')
    ap.add_argument('--max_cand', type=int, default=400)
    ap.add_argument('--report', type=str, default='runs/chosen_report.json')
    a = ap.parse_args()
    ck = torch.load(a.src, map_location='cpu', weights_only=False)
    buf = ck['buf']
    acc, mg = ck['acc'], ck['margin']
    cand = torch.nonzero(acc >= 0.99999).flatten()
    cand = cand[torch.argsort(mg[cand], descending=True)][:a.max_cand]
    print(f'{len(cand)} candidate members with exact held-out accuracy')

    best, n_cert = None, 0
    for i in cand.tolist():
        code = torch.cat([torch.zeros(1), ck['code'][i]])
        ok, rep = certify.certify(code, ck['knee'][i], float(ck['fold'][i]),
                                  bank_w=buf['bank_w'], key_w=buf['key_w'],
                                  val_w=buf['val_w'], lam=buf['lam'], verbose=False)
        n_cert += ok
        if ok and (best is None or rep['safety_factor'] > best[1]['safety_factor']):
            best = (i, rep)
    print(f'{n_cert} of them certify on the whole domain')
    if best is None:
        print('NO CERTIFIED MEMBER'); sys.exit(1)
    i, rep = best
    print(f'chose member {i}')
    for k, v in rep.items():
        print(f'  {k}: {v}')

    extra = ''.join('        %r: %r,\n' % kv for kv in [
        ('worst_case_readout_margin', round(rep['worst_readout_margin'], 6)),
        ('attention_notch_depth', round(rep['notch_depth'], 3)),
        ('certified_exact_on_whole_domain', True)])
    np_ = emit(ck['code'][i].tolist(), ck['knee'][i].tolist(), [float(ck['fold'][i])],
               buf, a.out, extra, s3=ck.get('stage3', False))
    print(f'wrote {a.out} with {np_} parameters')
    json.dump({'source': a.src, 'member': int(i), **rep}, open(a.report, 'w'), indent=1)


if __name__ == '__main__':
    main()
