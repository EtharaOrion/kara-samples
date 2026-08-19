#!/bin/sh
set -eu
install -d -m 0700 /logs/verifier
chmod -R go-rwx /tests

if [ ! -f /workspace/submission.py ] && [ -f /workspace/work/submission.py ]; then
  cp /workspace/work/submission.py /workspace/submission.py
fi
PYTHONPATH=/tests python -m adderboard_grader.grader \
  --submission /workspace/submission.py \
  --spec /tests/spec.json \
  --score-out /logs/verifier/score.json \
  --report-out /logs/verifier/report.json
if [ -f /tests/test_outputs.py ]; then
  python /tests/test_outputs.py
fi
