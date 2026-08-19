"""Unprivileged submission worker. It receives cases on stdin and returns JSON."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("adderboard_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct submission module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_count(model: Any, count_buffers: bool) -> int:
    tensors = list(model.parameters())
    if count_buffers:
        tensors.extend(model.buffers())
    storages: dict[tuple[str, int, int], int] = {}
    scalar_tensors = 0
    for tensor in tensors:
        if tensor is None:
            continue
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            element_size = max(1, int(tensor.element_size()))
            storages[key] = max(storages.get(key, 0), int(storage.nbytes()) // element_size)
        except (AttributeError, RuntimeError):
            scalar_tensors += int(tensor.numel())
    return sum(storages.values()) + scalar_tensors


ATTENTION_SOURCE_MARKERS = (
    "scaled_dot_product_attention",
    "multiheadattention",
    "softmax",
)


def _has_attention(model: Any, torch: Any, source: str = "") -> bool:
    """Whether the submission contains a self-attention computation.

    Name matching alone was wrong in both directions. It passes anything that
    calls a module "Attention" regardless of what that module does, and it fails
    genuine attention written inline. The second cost was real and expensive: a
    gpt-5.6-sol submission on 2026-08-13 computed
    `softmax(q @ k.T / sqrt(d)) @ v` directly inside a module named
    `AdditionTransformer`, reached 100 percent accuracy with perfect edges at
    179 parameters, and scored zero because no class name matched.

    Module names are still checked, since a named module is the common case, and
    the source is checked as well so an inline implementation counts. Whether
    that attention does any work is a separate question, answered by
    `_attention_is_degenerate`.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            return True
        name = module.__class__.__name__.lower()
        if "attention" in name or "attn" in name:
            return True
    lowered = (source or "").lower()
    if "scaled_dot_product_attention" in lowered:
        return True
    # A softmax over a matmul of two projections is the attention operation
    # itself, whatever the surrounding names are.
    return "softmax" in lowered and ("@" in lowered or "matmul" in lowered)


def _attention_is_degenerate(model: Any, torch: Any) -> bool:
    """True when the model's attention cannot respond to its input.

    Observed on 2026-08-11 across two independent gpt-5.6-sol rollouts: a module
    named SelfAttention that sets queries and keys to zeros, so the softmax is
    uniform no matter what arrives, and the operation is a plain mean. Scaling
    that mean by a single learned constant turns it into a sum, which is enough
    to hand-code decimal addition around while claiming a one-parameter
    transformer. Every other check passes on it, because the model really is
    called and really does determine the output.

    What separates attention from a fixed reduction is that its weights depend
    on its input. This probes exactly that: for a fixed input shape, feed
    several value patterns that are not multiples of one another and compare the
    ratio of the module's first output to the input sum. A fixed reduction gives
    the same ratio every time. Genuine attention re-weights, so the ratio moves.

    The token count is held constant per group, because the ratio of an honest
    mean legitimately depends on how many tokens it averages, and comparing
    across lengths would flag every fixed-length reduction.
    """
    modules = [
        module
        for module in model.modules()
        if module is not model
        and (
            isinstance(module, torch.nn.MultiheadAttention)
            or "attention" in module.__class__.__name__.lower()
            or "attn" in module.__class__.__name__.lower()
        )
    ]
    if not modules:
        return False

    # Deliberately not scalar multiples of each other: a linear attention with
    # no softmax would be scale invariant and would look degenerate under
    # proportional probes alone.
    patterns = (
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 9.0],
        [9.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 7.0],
    )

    for module in modules:
        ratios = []
        for values in patterns:
            probe64 = torch.tensor(values, dtype=torch.float64).reshape(1, 4, 1)
            for dtype in (torch.float64, torch.float32):
                try:
                    with torch.no_grad():
                        result = module(probe64.to(dtype))
                    if isinstance(result, tuple):
                        result = result[0]
                    total = float(probe64.sum())
                    if abs(total) < 1e-12:
                        break
                    ratios.append(float(result.to(torch.float64).flatten()[0]) / total)
                    break
                except Exception:  # noqa: BLE001
                    continue
        # Fewer than three usable probes means this module could not be
        # exercised, which is not evidence against it.
        if len(ratios) >= 3 and (max(ratios) - min(ratios)) < 1e-6:
            return True
    return False


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    submission_path = Path(payload["submission"])
    with contextlib.redirect_stdout(sys.stderr):
        module = _load(submission_path)
        if not hasattr(module, "build_model") or not hasattr(module, "add"):
            raise ValueError("submission must define build_model() and add()")
        built = module.build_model()
        if not isinstance(built, tuple) or len(built) != 2:
            raise ValueError("build_model() must return (model, metadata)")
        model, metadata = built
        if not isinstance(model, torch.nn.Module):
            raise TypeError("build_model() must return a torch.nn.Module")
        if not isinstance(metadata, dict):
            raise TypeError("build_model() metadata must be a dictionary")

        call_counter = [0]

        def observed_call(_module: Any, _args: Any) -> None:
            call_counter[0] += 1

        hook = model.register_forward_pre_hook(observed_call)
        outputs: list[Any] = []
        model_calls: list[int] = []
        try:
            for a, b in payload["cases"]:
                before = call_counter[0]
                result = module.add(model, int(a), int(b))
                outputs.append(result if isinstance(result, int) else repr(result))
                model_calls.append(call_counter[0] - before)
        finally:
            hook.remove()

        dependence_rate = _output_dependence(module, model, payload.get("probe_cases", []))

    return {
        "outputs": outputs,
        "model_calls": model_calls,
        "dependence_rate": dependence_rate,
        "params": _tensor_count(model, bool(payload.get("count_buffers", False))),
        "has_attention": _has_attention(model, torch, payload.get("source", "")),
        "degenerate_attention": _attention_is_degenerate(model, torch),
        "model_type": model.__class__.__name__,
        "claimed_params": metadata.get("params"),
    }


def _output_dependence(module: Any, model: Any, probe_cases: list) -> float:
    """Fraction of probe answers that change when the model output is corrupted.

    A genuine solver reads its answer off the model output, so replacing every
    forward result with zeros must move the answer on essentially every probe.
    A submission that calls forward only to satisfy the liveness counter and
    computes the sum in Python leaves its answers unchanged, which this gate
    catches without any task-specific knowledge of addition.
    """
    if not probe_cases:
        return 0.0

    def corrupt(_module: Any, _inputs: Any, output: Any):
        try:
            return output * 0
        except (TypeError, ValueError):
            return output

    unchanged = 0
    total = 0
    hook = model.register_forward_hook(corrupt)
    try:
        for a, b in probe_cases:
            try:
                clean = module.add(model, int(a), int(b))
            except Exception:
                return 0.0
    finally:
        hook.remove()
    # Compare clean and corrupted answers over the same probe set.
    clean_answers = []
    for a, b in probe_cases:
        try:
            clean_answers.append(module.add(model, int(a), int(b)))
        except Exception:
            return 0.0
    hook = model.register_forward_hook(corrupt)
    try:
        for (a, b), clean in zip(probe_cases, clean_answers):
            try:
                corrupted = module.add(model, int(a), int(b))
            except Exception:
                return 0.0
            total += 1
            if corrupted == clean:
                unchanged += 1
    finally:
        hook.remove()
    if total == 0:
        return 0.0
    return 1.0 - unchanged / total


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        response = {"ok": True, "result": run(payload)}
    except BaseException as exc:
        response = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    json.dump(response, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()

