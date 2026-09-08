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

from .operations import resolve
from .policy import inspect_source
from .score import DEFAULT_PARAMETER_FLOOR, bounded_score


def build_cases(spec: dict[str, Any]) -> tuple[list[tuple[int, ...]], int]:
    operation = resolve(spec.get("operation"))
    digits = int(spec["digits"])
    maximum = 10**digits - 1
    # Every fixed case goes through the same domain map as the random ones, so
    # a family whose domain is a strict subset of the operand space cannot ship
    # an unreachable edge case at a narrow digit width.
    edges = [operation.order(*case) for case in operation.edge_cases(maximum)]
    rng = random.Random(int(spec["seed"]))
    # Each family draws its own admitted domain. The classic families' default
    # draw makes exactly the rng call sequence this function made before the
    # draw method existed, so every frozen seed keeps its original case list.
    random_cases = [
        operation.draw(rng, maximum)
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


def _probe_cases(spec: dict[str, Any]) -> list[tuple[int, ...]]:
    """A small fixed set of non-trivial operand tuples for the dependence gate.

    Drawn from a probe-specific seed so it never overlaps the graded set, and
    kept off any degenerate answer so a corrupted-to-zero model output genuinely
    moves the answer for a real solver.
    """
    operation = resolve(spec.get("operation"))
    maximum = 10 ** int(spec["digits"]) - 1
    rng = random.Random(int(spec["seed"]) ^ 0x9E3779B9)
    return [operation.probe_pair(rng, maximum) for _ in range(24)]


def _run_worker(
    submission: Path,
    cases: list[tuple[int, ...]],
    timeout: float,
    count_buffers: bool,
    probe_cases: list[tuple[int, ...]] | None = None,
    entrypoint: str = "add",
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
        "entrypoint": entrypoint,
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
    operation = resolve(spec.get("operation"))
    findings = inspect_source(submission, operation.entrypoint, operation.arity)
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
                operation.entrypoint,
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"

    outputs = worker_result.get("outputs", [])
    calls = worker_result.get("model_calls", [])
    correct = 0
    edge_correct = 0
    if len(outputs) == len(cases):
        for index, (case, output) in enumerate(zip(cases, outputs)):
            is_correct = isinstance(output, int) and output == operation.apply(*case)
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
    has_attention = bool(worker_result.get("has_attention"))
    structural_failures = []
    if findings:
        structural_failures.append("static_policy")
    if error is not None:
        structural_failures.append("worker_error")
    if not has_attention:
        structural_failures.append("attention_not_detected_or_not_output_dependent")
    if len(outputs) != len(cases):
        structural_failures.append("incomplete_outputs")
    if not output_dependent:
        structural_failures.append("model_output_not_used")
    structurally_valid = not structural_failures
    # A structural failure otherwise scores zero with nothing said about why,
    # which reads to the agent as "the submission was fine and the grader broke".
    # The instruction promises it will be told what was objected to, so each
    # failure that is not already a static finding is stated in the agent's
    # terms. Attention is the one worth spelling out: a module can exist, be
    # called, and still be a fixed reduction wearing the name, and two
    # independent rollouts reached a perfect score exactly that way.
    _FAILURE_FINDINGS = {
        "attention_not_detected_or_not_output_dependent": (
            "no attention operation was observed doing work: either none ran, or "
            "perturbing it left the answers unchanged, which makes it a fixed "
            "reduction rather than attention"
        ),
        "model_output_not_used": (
            "the returned answers did not change when the model output was "
            "corrupted, so the answer is not being read off the model"
        ),
        "incomplete_outputs": "the submission did not return an answer for every case",
    }
    findings = sorted(
        set(findings)
        | {
            _FAILURE_FINDINGS[failure]
            for failure in structural_failures
            if failure in _FAILURE_FINDINGS
        }
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
        "operation": operation.name,
        "upstream_commit": spec.get("upstream_commit"),
        "structurally_valid": structurally_valid,
        "findings": findings,
        "error": error,
        "model_type": worker_result.get("model_type"),
        "claimed_params": worker_result.get("claimed_params"),
        "measured_params": params,
        "has_attention": has_attention,
        "attention_probe": worker_result.get("attention"),
        "structural_failures": structural_failures,
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
