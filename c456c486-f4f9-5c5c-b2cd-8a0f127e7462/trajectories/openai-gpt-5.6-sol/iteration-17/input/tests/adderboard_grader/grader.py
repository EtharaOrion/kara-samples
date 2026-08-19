"""Hardened Harbor verifier for an AdderBoard submission."""

from __future__ import annotations

import argparse
import json
import math
import os
import pwd
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .policy import inspect_source
from .score import DEFAULT_PARAMETER_FLOOR, bounded_score


def _edge_cases(maximum: int) -> list[tuple[int, int]]:
    half = maximum // 2
    return [
        (0, 0),
        (0, 1),
        (1, 0),
        (maximum, 0),
        (maximum, 1),
        (maximum, maximum),
        (half, maximum - half),
        (maximum // 9, maximum - maximum // 9),
        (1, maximum),
        (maximum - 1, 1),
    ]


def build_cases(spec: dict[str, Any]) -> tuple[list[tuple[int, int]], int]:
    digits = int(spec["digits"])
    maximum = 10**digits - 1
    edges = _edge_cases(maximum)
    rng = random.Random(int(spec["seed"]))
    random_cases = [
        (rng.randint(0, maximum), rng.randint(0, maximum))
        for _ in range(int(spec["random_tests"]))
    ]
    return edges + random_cases, len(edges)


def _drop_privileges() -> Callable[[], None] | None:
    if os.name != "posix" or os.geteuid() != 0:
        return None
    nobody = pwd.getpwnam("nobody")

    def drop() -> None:
        os.setgroups([])
        os.setgid(nobody.pw_gid)
        os.setuid(nobody.pw_uid)

    return drop


def _probe_cases(spec: dict[str, Any]) -> list[tuple[int, int]]:
    """A small fixed set of non-trivial pairs for the output-dependence gate.

    Drawn from a probe-specific seed so it never overlaps the graded set, and
    kept off any zero operand so a corrupted-to-zero model output genuinely
    moves the answer for a real solver.
    """
    maximum = 10 ** int(spec["digits"]) - 1
    rng = random.Random(int(spec["seed"]) ^ 0x9E3779B9)
    return [(rng.randint(1, maximum), rng.randint(1, maximum)) for _ in range(24)]


def _run_worker(
    submission: Path,
    cases: list[tuple[int, int]],
    timeout: float,
    count_buffers: bool,
    probe_cases: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    worker_source = Path(__file__).with_name("worker.py")
    runtime_dir = Path(tempfile.mkdtemp(prefix="adderboard-worker-", dir="/tmp"))
    os.chmod(runtime_dir, 0o755)
    worker_path = runtime_dir / "worker.py"
    shutil.copyfile(worker_source, worker_path)
    os.chmod(worker_path, 0o555)

    payload = {
        "submission": str(submission),
        "cases": cases,
        "count_buffers": count_buffers,
        "probe_cases": probe_cases or [],
        # The worker inspects the source as well as the built model, because
        # attention written inline has no module to find by name.
        "source": submission.read_text(encoding="utf-8", errors="replace"),
    }
    env = {
        "HOME": str(runtime_dir),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    try:
        result = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(payload, separators=(",", ":")),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            preexec_fn=_drop_privileges(),
            check=False,
        )
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
    if result.returncode:
        raise RuntimeError(
            f"submission worker exited {result.returncode}: {result.stderr[-2000:]}"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"submission worker returned invalid JSON: {result.stdout[-1000:]}"
        ) from exc
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "submission worker failed"))
    return response["result"]


def evaluate(submission: Path, spec: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    started = time.monotonic()
    findings = inspect_source(submission)
    cases, edge_count = build_cases(spec)
    worker_result: dict[str, Any] = {}
    error: str | None = None

    if not findings:
        try:
            worker_result = _run_worker(
                submission,
                cases,
                float(spec.get("worker_timeout_sec", 240)),
                bool(spec.get("count_buffers", False)),
                _probe_cases(spec),
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"

    outputs = worker_result.get("outputs", [])
    calls = worker_result.get("model_calls", [])
    correct = 0
    edge_correct = 0
    if len(outputs) == len(cases):
        for index, ((a, b), output) in enumerate(zip(cases, outputs)):
            is_correct = isinstance(output, int) and output == a + b
            correct += int(is_correct)
            if index < edge_count:
                edge_correct += int(is_correct)

    accuracy = correct / len(cases)
    edge_accuracy = edge_correct / edge_count
    model_use_rate = (
        sum(int(isinstance(value, int) and value > 0) for value in calls) / len(cases)
        if len(calls) == len(cases)
        else 0.0
    )
    params = int(worker_result.get("params", 0) or 0)
    dependence_rate = float(worker_result.get("dependence_rate", 0.0) or 0.0)
    output_dependent = dependence_rate >= 0.9
    # An attention module whose weights do not respond to its input is a fixed
    # reduction wearing the name. Two independent rollouts reached a perfect
    # score that way: zero queries and keys make the softmax uniform, one learned
    # constant turns the resulting mean into a sum, and decimal addition is
    # hand-coded around it. Every other gate passes on that submission, so this
    # is the one that has to stop it, and it is reported as a finding rather than
    # a silent zero so the agent learns what was objected to.
    if bool(worker_result.get("degenerate_attention")):
        findings = sorted(
            set(findings)
            | {
                "attention weights do not depend on the input, so the module is a "
                "fixed reduction rather than attention"
            }
        )
    structurally_valid = (
        not findings
        and error is None
        and bool(worker_result.get("has_attention"))
        and len(outputs) == len(cases)
        and output_dependent
    )
    score, efficiency, qualified = bounded_score(
        structurally_valid=structurally_valid,
        accuracy=accuracy,
        edge_accuracy=edge_accuracy,
        model_use_rate=model_use_rate,
        params=params,
        threshold=float(spec["accuracy_threshold"]),
        baseline=int(spec["baseline_parameters"]),
        floor=int(spec.get("floor_parameters", DEFAULT_PARAMETER_FLOOR)),
        worst=int(spec.get("worst_parameters", 0)) or None,
    )
    target = int(spec.get("target_parameters", 0) or 0)
    elapsed = time.monotonic() - started
    scores = {
        "score": float(score),
        "accuracy": float(accuracy),
        "edge_accuracy": float(edge_accuracy),
        "qualified": float(qualified),
        "model_use_rate": float(model_use_rate),
        "parameter_efficiency": float(efficiency),
        "parameters": float(params),
        "output_dependence": float(dependence_rate),
        "elapsed_seconds": float(elapsed),
    }
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError("grader produced a non-finite metric")
    report = {
        "task": spec.get("task"),
        "upstream_commit": spec.get("upstream_commit"),
        "structurally_valid": structurally_valid,
        "degenerate_attention": bool(worker_result.get("degenerate_attention")),
        "findings": findings,
        "error": error,
        "model_type": worker_result.get("model_type"),
        "claimed_params": worker_result.get("claimed_params"),
        "measured_params": params,
        # target_parameters no longer anchors the size score -- it is a reported
        # milestone so the curriculum keeps a human-readable "good model here"
        # marker without putting a flat spot in the score.
        "target_parameters": target,
        "beat_target": bool(target and 0 < params <= target),
        "output_dependence": dependence_rate,
        "output_dependent": output_dependent,
        "passed": correct,
        "total": len(cases),
        "metrics": scores,
    }
    return scores, report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--score-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    try:
        scores, report = evaluate(args.submission.resolve(), spec)
    except BaseException as exc:  # never leave the score path unwritten
        reason = f"grader failed: {type(exc).__name__}: {exc}"
        scores = {
            "score": 0.0, "accuracy": 0.0, "edge_accuracy": 0.0, "qualified": 0.0,
            "model_use_rate": 0.0, "parameter_efficiency": 0.0, "parameters": 0.0,
            "output_dependence": 0.0, "elapsed_seconds": 0.0,
        }
        report = {
            "task": spec.get("task"), "structurally_valid": False,
            "findings": [], "error": reason, "metrics": scores,
        }
        _write_json(args.report_out, report)
        _write_json(args.score_out, scores)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    _write_json(args.report_out, report)
    _write_json(args.score_out, scores)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
