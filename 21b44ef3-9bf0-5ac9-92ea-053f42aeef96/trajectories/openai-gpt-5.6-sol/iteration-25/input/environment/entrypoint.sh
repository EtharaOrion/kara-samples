#!/bin/sh
set -eu

# No symlink here on purpose. This used to be `ln -s /workspace /workspace/work`
# so that agents which invent an intermediate `work/` directory still wrote to
# the graded path. It worked in the container and broke everything downstream:
# Harbor copies /workspace out verbatim, so the collected artifacts carried an
# absolute link to a path that does not exist on the host. A dangling,
# self-referential symlink is not an artifact.
#
# The fallback now lives in the verifier's test.sh, which copies
# /workspace/work/submission.py into place if the graded path is empty. That is
# a real file operation with no link, and it happens where it matters: at
# grading time.

exec "$@"
