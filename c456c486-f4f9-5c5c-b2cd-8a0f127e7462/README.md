# `c456c486-f4f9-5c5c-b2cd-8a0f127e7462` - `Adder Board`

Two model cohorts (`claude-opus-5`, `openai/gpt-5.6-sol`), one refinement session each, 56 completed iterations packaged.

The task is the smallest trained transformer that adds two 14-digit integers exactly, at 99% held-out accuracy or better. The score is a bounded continuous function of parameter count, so the objective has headroom the whole way down rather than a target to hit and stop at.

```
task                    arith/add-14d-trained
spec digest             sha256:62d4f97b4af6635319802ed8d392f7a8e043bb2ab1c7dde47508d0d286d50590
image                   426628337772.dkr.ecr.ap-south-1.amazonaws.com/kara:kara-arith-rl_v1
                        @sha256:d6ff33b01911559ed76cf4d52df47d489e89e4ac34e25b6854c9c403e6ef9550
verify the spec         sha256sum dataset/tests/spec.json  (spec_digest covers the graded contract)
```

`dataset/` is the frozen task exactly as graded. The directory name is `uuid5` over that bundle's canonical content, so the name is its own integrity check — but note this delivery reorganises the bundle under `dataset/` and adds `trajectories/`, so the uuid reproduces against the original freeze, not against this directory as laid out here.

## Session result

| | `claude-opus-5` | `openai/gpt-5.6-sol` |
|---|---|---|
| iterations packaged | **6 of 50** | **50 of 50** |
| iterations counted correct | 5 | 34 |
| selection | `best` → **iteration 6** | `best` → **iteration 42** |
| best score | **1.000000** | **0.595851** |
| smallest qualifying model | **7 parameters** | 1,266 parameters |
| its accuracy / edge accuracy | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| stopped because | full score with every rubric passed | iteration cap reached |
| total tokens | 69,481,949 | 69,685,988 |

`claude-opus-5` reached 7 parameters — one below the point where the score function saturates at 1.0 — by tokenising the operands in base 2 so the digit-embedding table shrinks from ten scalars to two, then folding parameters through exact reparametrisations. Its descent was 154 → 115 → 33 → 21 → 7 across five qualifying iterations.

## Conformance to the client specification

| requirement | state |
|---|---|
| `instruction.md` states an open-ended objective | pass — "build the smallest trained transformer that reaches at least 99.00% held-out accuracy… keep making it smaller while it still meets the accuracy bar" |
| a sub-optimal implementation to improve on | pass — `dataset/environment/submission.py` ships a deliberately weak baseline with its trainer alongside as `train.py` |
| score is a float in [0,1], never binary | pass — `(0.5 + 0.5·size_efficiency) · accuracy_factor · edge_factor`, piecewise-log in parameter count between a floor of 8 and a worst case of 16,440 |
| one H100 | pass — the agent container receives exactly one GPU (`torch.cuda.device_count() == 1`), delivered by a compose device reservation because the runner's Docker backend refuses a non-zero `gpus` count in the manifest |
| `task.toml` tags `optimization` and `bounded-continuous` | pass — both present in `keywords` |
| `tests/rubrics.jsonl` plus an LLM trajectory grader | pass — 8 atomic rubrics, one model call each, verdicts and rationales in every `iteration-NN/output/rubric_judgments.json` |
| only a solution where all rubrics pass counts as correct | pass — `counts_as_correct = met_accuracy_bar AND every rubric passed`; `effective_score` is 0 for any iteration the review rejected |
| refinement loop: summarise, re-attempt, fixed count, pick best | pass — every iteration's `input/instruction.md` carries the record of all prior iterations with their scores; `iteration-01` has none by construction |
| an HTML file to inspect each task | pass — `inspector.html` |
| two plots, x-axis iterations and x-axis total tokens | pass — inside `inspector.html`, plus per-model score-per-iteration and parameter-descent charts |
| full internet for setup, agent installs what it needs | the agent environment is `network_mode = "public"` for the whole attempt, not setup only. The verifier is `no-network`, so grading is sealed |
| `max_timeout` in hours, capped at N attempts | pass — `max_timeout = 72.0`, `N = 50`; the budget is a pool shared across iterations rather than a per-iteration ration |
| each task solvable within 6 hours of one H100 | **met in practice, over-declared** — `claude-opus-5` produced a qualifying 154-parameter solution in its first 6.7 h iteration and the winning 7-parameter model in 2.7 h; but `max_timeout = 72.0` declares a budget twelve times the stated bound |

## Caveats you should read before quoting a number

**Scores are gated, and the raw measurement is kept beside them.** `score` is what the deterministic verifier measured; `effective_score` applies the conduct gate and is 0 for any iteration the review rejected. Rank on `effective_score`. The two differ exactly where a submission measured well and was caught — `openai/gpt-5.6-sol` iteration 1 measured **0.9398** on a 17-parameter model that propagated carries in host code, and contributes **0.0**. Quoting its raw 0.9398 as a result would be quoting a rejected submission.

**The agents' own summaries are unverified.** Each iteration's `approach` and `what happened` are written by a summariser reading the trajectory, not measured. They have been wrong: one summary claimed "10,000/10,000 unseen pairs" for a run whose loop had been cut to 2,000, and another wrote up a conduct-rejected submission as a success. Where a summary disagrees with `output/jobs/verifier/score.json`, the verifier is correct.

**The oracle ships in this bundle.** `dataset/solution/` contains `reference_submission.py` — an 11-parameter model that grades **0.9746** — and `TRUTH.md`, which sets out the solve path step by step. Anyone holding this directory can reproduce a top-ranked result directly, so the task cannot be used to evaluate a model that has had access to it.

**Two iterations produced no graded result.** `claude-opus-5` iteration 2 and `openai/gpt-5.6-sol` iteration 36 failed before submitting; their `output/jobs/result.json` carries the runner's traceback. Their artifact rubrics record `error`, not `fail`, because an iteration with no submission has nothing to judge and scoring that as a violation would manufacture a false accusation.

**Paths inside the delivered files are relative to this directory,** rewritten from the absolute run-host paths they were recorded with. Two `result.json` files still carry absolute paths inside Python tracebacks; those are stack frames of the failure, not artifact locations, and were left as recorded.

**The vocabulary is "score" throughout.** The harness records this quantity as a reward and the term was renamed for delivery, including inside the agents' transcripts and the history blocks they were shown. The originals on the run host retain the earlier wording.

## Files

```
README.md                   this file
inspector.html              browsable view: task contract, why it is hard, where the
                            models fail, every agent-visible file, and each iteration
                            opened into verifier / conduct / counters / agent account
iterations.json             flat per-iteration index: score, gated score, accuracy,
                            parameters, rubric outcomes, tokens, and input/output paths
dataset/                    the frozen task, exactly as graded
  instruction.md            the objective handed to the agent
  task.toml                 manifest: budget, image digest, GPU, network mode
  environment/              starting files, including the weak baseline and its trainer
  tests/                    spec.json, rubrics.jsonl, and the grader that produced
                            every score in this delivery
  solution/                 PRIVATE — reference solution and solve path (see caveats)
trajectories/<model>/
  attempts.json             the run ledger, one record per iteration
  final_submission.py       the submission the declared selection rule chose
  final_submission.json     which iteration it came from, and why
  iteration-NN/
    input/                  the task copy this iteration was handed; its
                            instruction.md carries every prior iteration and score
    output/
      rubric_judgments.json all 8 conduct verdicts with the judge's rationale
      weights/              the graded model's parameters, rebuilt as the grader
                            builds them, plus a readable tensor summary
      jobs/
        config.json         the trial as configured
        result.json         the trial as it ended
        verifier/           score.json and report.json — the authority on this run
        artifacts/          the graded submission.py and the collected workspace
        agent/              the transcript, the structured trajectory, the session
                            record, and the per-call completions
```
