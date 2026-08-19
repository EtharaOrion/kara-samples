## Ordered path

1. Read `instruction.md`; the single graded outcome is the score float. Satisfies checker `score_schema`. Deliverable `D1`.
2. Place `solution/reference_submission.py` at `/workspace/submission.py` unchanged; the established state is an eleven-parameter trained transformer with a documented interval-bound exactness argument. Satisfies checker `static_policy`, since the file imports only allowed roots and computes through the model.
3. The verifier imports the file, recounts parameters from tensors, and observes a forward call on every case. Satisfies checkers `parameter_recount` and `model_use_rate`.
4. Hidden evaluation over the rotated frozen seed grades all ten thousand random pairs and every fixed edge case exactly. Satisfies checkers `accuracy` and `edge_accuracy`.
5. The verifier writes `/logs/verifier/score.json` with only finite numeric fields, score exactly the bound full-score value one. Satisfies red-line `RL1`.

## Rejected routes

- Direct arithmetic disguised as a model: measured rejected by the `shortcut fixture rejected by static policy` control.
- Submitting the unchanged template: measured rejected by the `empty template scores exactly zero` control.
- A structurally valid but untrained model: measured non-qualified by the `constant-output model earns no qualification` control.

## Checker closure

Named checkers: accuracy, edge_accuracy, model_use_rate, parameter_recount, static_policy, score_schema. Deliverables: D1. Red-lines: RL1. These sets must equal the committed grader surface, and Phase 2 reconciles them by identifier.
