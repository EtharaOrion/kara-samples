"""Training pipeline for the model in submission.py: random init -> 33 parameters.

Nothing here is imported by the graded file; export.py is what writes a trained
checkpoint into submission.py as weight literals.

The shape of the search
-----------------------
Finding the carry-lookahead circuit from a random init is a lottery: roughly one
init in eight lands in it, the rest settle into an analogue-carry basin that
saturates around 80% exact match.  So stage 1 (lab_b2.py) trains a *population*
of models at once -- every parameter carries a leading "model" dimension, so 16
models cost about what one costs -- and re-initialises cells that are both bad
and past a fair trial length.

Every later stage is one structural cut applied to a converged model, followed
by re-training the whole thing under the cut (lab_ft.py, again as a population,
this time of jittered copies of the warm start).  Warm starting only chooses
where training begins: every weight in the shipped file is one that gradient
descent put there, and each stage was trained until it scored 1.00000 exact
match on 10^6 held-out uniform pairs and 10^6 held-out carry-stress pairs
before the next cut was attempted.

Some cuts are exact re-parametrisations -- the model computes the same function
with fewer parameters, and training afterwards only cleans up what the fold
approximated.  Those are marked "exact" below and are derived in the module
named on each line.

Cuts that were tried and did not hold the accuracy bar are recorded at the
bottom; they are the evidence that the remaining parameters are load-bearing.
"""
import argparse, subprocess, sys

PY = sys.executable

STAGES = [
    # (parameters, what the cut is, command)
    (74, 'stage 1: population from scratch, 5+4 hidden units, free readout',
     [PY, 'lab_b2.py', '--steps', '80000', '--out', 'ckpt_b2.pt']),

    (62, 'tie readout column 0 to the digit code U, and drop the query and value '
         'scales (exact, lab_e.fold)',
     [PY, 'lab_ft.py', '--src', 'ckpt_b2.pt', '--index', '1', '--arch', 'e', '--tied',
      '--d_ffa', '5', '--d_ffb', '4', '--steps', '30000', '--lr', '1.5e-3',
      '--out', 'ckpt_ft_e62.pt']),

    (58, 'write the output MLP into the digit-code channel only',
     [PY, 'lab_ft.py', '--src', 'ckpt_ft_e62.pt', '--arch', 'f', '--tied',
      '--d_ffa', '5', '--d_ffb', '4', '--steps', '30000', '--lr', '1.5e-3',
      '--out', 'ckpt_ft_f58a.pt']),

    (54, 'narrow the output MLP to 3 units',
     [PY, 'lab_ft.py', '--src', 'ckpt_ft_f58a.pt', '--index', '15', '--arch', 'f', '--tied',
      '--d_ffa', '5', '--d_ffb', '3', '--steps', '30000', '--lr', '2e-3',
      '--out', 'ckpt_ft_f54b.pt']),

    (50, 'narrow the output MLP to 2 units',
     [PY, 'lab_ft.py', '--src', 'ckpt_ft_f54b.pt', '--index', '1', '--arch', 'f', '--tied',
      '--d_ffa', '5', '--d_ffb', '2', '--steps', '30000', '--lr', '2.5e-3',
      '--out', 'ckpt_ft_f50.pt']),

    (47, 'trade the feature MLP\'s constant unit for one scalar (exact, lab_g.fold)',
     [PY, 'lab_ft.py', '--src', 'ckpt_ft_f50.pt', '--arch', 'g', '--tied',
      '--d_ffa', '5', '--d_ffb', '2', '--steps', '30000', '--lr', '1.5e-3',
      '--out', 'ckpt_ft_g47.pt']),

    (43, 'drop a redundant feature unit by re-solving the key and value read-outs '
         'on the remaining three (exact on the digit-sum lattice, refit.py)',
     [PY, 'refit.py', '--ckpt', 'ckpt_ft_g47.pt', '--keep', '3', '--out', 'ckpt_refit43.pt']),
    (43, '  ... and re-train in the narrower parametrisation',
     [PY, 'lab_ft.py', '--src', 'ckpt_refit43.pt', '--arch', 'g', '--tied',
      '--d_ffa', '3', '--d_ffb', '2', '--steps', '30000', '--lr', '6e-4', '--seed', '7',
      '--jitters', '0,0.01,0.03,0.08', '--out', 'ckpt_ft_g43.pt']),

    (36, 'decode by distance to the digit code instead of a learned readout column, '
         'which also makes the value offset redundant (lab_h)',
     [PY, 'lab_ft.py', '--src', 'ckpt_ft_g43.pt', '--arch', 'h', '--tied',
      '--d_ffa', '3', '--d_ffb', '3', '--steps', '40000', '--lr', '2e-3', '--seed', '32',
      '--reinit_out', '--jitters', '0,0.03,0.08,0.2', '--out', 'ckpt_ft_h36.pt']),

    (33, 'give the head a scalar output projection, replacing the output MLP unit '
         'that only passed the carry through (exact, lab_i.fold)',
     [PY, 'fold33.py', '--src', 'ckpt_ft_h36.pt', '--out', 'ckpt_fold33.pt']),
    (33, '  ... and re-train in the narrower parametrisation',
     [PY, 'lab_ft.py', '--src', 'ckpt_fold33.pt', '--arch', 'i', '--tied',
      '--d_ffa', '3', '--d_ffb', '2', '--steps', '30000', '--lr', '4e-4', '--seed', '51',
      '--sigma', '0.015', '--jitters', '0,0.01,0.03,0.08', '--out', 'ckpt_ft_i33.pt']),

    (33, 'write the weights into the graded file',
     [PY, 'export.py', '--ckpt', 'ckpt_ft_i33.pt', '--arch', 'i', '--out', 'submission.py']),
]

REJECTED = """Cuts that did not hold the bar (loss they stalled at, cross-entropy):
   4 feature units instead of 5 by pruning, before the refit made it exact  0.7-1.7
   2 feature units: no such model exists -- with an additive digit code and a
     ReLU layer, a key that dips on digit sums of 9 needs three units, and least
     squares on the 19-point digit-sum lattice misses by 2.5 of a 3.1 notch      --
   1 output MLP unit                                                          1.78
   output MLP without its biases                                              0.21
   output MLP writing into the carry channel instead of the digit channel      0.50
   no readout column for the carry, output MLP of 2 units                      1.95
   no readout column for the carry, output MLP of 3 units                      1.63
     (both of those are the same failure: with one readout direction the logits
      are linear in the class code, so the argmax is always the largest or
      smallest digit.  The distance readout at 36 fixes it for free.)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true', help='actually run the pipeline')
    ap.add_argument('--from_stage', type=int, default=0)
    args = ap.parse_args()

    for i, (n, what, cmd) in enumerate(STAGES):
        print(f'[{i}] {n:3d} params  {what}\n      {" ".join(cmd)}')
        if args.run and i >= args.from_stage:
            subprocess.run(cmd, check=True, cwd='/workspace')
    print()
    print(REJECTED)


if __name__ == '__main__':
    main()
