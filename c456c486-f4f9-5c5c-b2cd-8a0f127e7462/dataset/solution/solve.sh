#!/bin/sh
# Oracle: place the reference solution at the graded workspace path.
# canary 70312a2a50d2926c72d313ea066049a0
set -eu
cp "$(dirname "$0")/reference_submission.py" /workspace/submission.py
