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


def _looks_like_attention_scores(value: Any, dim: Any) -> bool:
    """Whether a softmax input has the shape of a batched score matrix.

    Deliberately does not require the matrix to be square. Requiring it reads
    only self-attention: a cross-attention block scores N queries against M
    keys, and rejecting that shape would fail a legitimate architecture for
    having the wrong proportions. Shape alone is therefore weak evidence, and
    the monitor confirms it by checking the softmax result is consumed by a
    matmul - which is what makes the numbers attention weights rather than a
    distribution over classes.
    """
    try:
        rank = int(value.dim())
        shape = tuple(int(size) for size in value.shape)
        axis = rank - 1 if dim is None else int(dim) % rank
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return False
    return rank >= 3 and axis == rank - 1 and shape[-1] > 1


class _AttentionOpMonitor:
    """Observe and optionally perturb attention operations during one forward pass.

    Known gap, deliberately left: this verifies the answer depends on the
    attention *values*, not that those values depend on the *input*. A learned
    constant score matrix - a fixed non-uniform pattern that ignores its input -
    survives perturbation and would pass, and the instruction text forbids
    exactly that. Every hack observed so far has been the uniform variant, which
    is caught here because perturbing uniform to uniform changes nothing. Closing
    the gap properly means comparing scores across differing inputs.

    A class name is not evidence of attention: it creates false negatives for
    inline/custom heads and false positives for unused decoy modules. Runtime
    observation accepts the common custom implementations (`torch.softmax`,
    `F.softmax`, or `Tensor.softmax` over a batched square score matrix) and the
    fused scaled-dot-product primitive.
    """

    def __init__(self, torch: Any, perturb: bool = False):
        self.torch = torch
        self.perturb = perturb
        self.mechanisms: set[str] = set()
        self._originals: list[tuple[Any, str, Any]] = []
        # Softmax results that had the shape of scores, held by reference so
        # their id() cannot be recycled onto an unrelated tensor mid-pass.
        self._scores: list[Any] = []
        self._score_ids: set[int] = set()

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name, None)
        if not callable(original):
            return
        try:
            setattr(owner, name, replacement(original))
        except (AttributeError, TypeError):
            return
        self._originals.append((owner, name, original))

    def _softmax_wrapper(self, original: Any):
        def observed(value: Any, *args: Any, **kwargs: Any):
            result = original(value, *args, **kwargs)
            dim = kwargs.get("dim", args[0] if args else None)
            if _looks_like_attention_scores(value, dim):
                # Shape only. The mechanism is claimed when this result is fed
                # to a matmul; see _matmul_wrapper.
                self._scores.append(result)
                self._score_ids.add(id(result))
                if self.perturb:
                    axis = (value.dim() - 1 if dim is None else int(dim) % value.dim())
                    return self.torch.full_like(result, 1.0 / int(result.shape[axis]))
            return result

        return observed

    def _weighting_wrapper(self, original: Any):
        """Confirm a score-shaped softmax result is used to weight values.

        Softmax shape alone does not distinguish attention from a classifier
        head: a digit distribution over ten classes is also a rank-3 softmax
        over the last axis, and it is trivially output-dependent, so accepting
        shape alone would reduce this gate to "calls softmax somewhere". What
        makes the numbers attention weights is that they multiply values.

        Both spellings count, because both appear in real submissions:
        `weights @ values` and the broadcast form `(weights * values).sum(-1)`.
        A classifier head is normally argmaxed rather than multiplied, so it
        still does not qualify. A submission that multiplies a logits softmax
        by something to manufacture the link must additionally survive the
        perturbation pass, which replaces those weights with a uniform
        distribution and requires the answers to move.
        """

        def observed(*args: Any, **kwargs: Any):
            if any(id(arg) in self._score_ids for arg in args):
                self.mechanisms.add("score_softmax_weighting")
            return original(*args, **kwargs)

        return observed

    def _sdpa_wrapper(self, original: Any):
        def observed(*args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            self.mechanisms.add("scaled_dot_product_attention")
            return self.torch.zeros_like(result) if self.perturb else result

        return observed

    def __enter__(self):
        self._patch(self.torch, "softmax", self._softmax_wrapper)
        functional = getattr(getattr(self.torch, "nn", None), "functional", None)
        if functional is not None:
            self._patch(functional, "softmax", self._softmax_wrapper)
            self._patch(
                functional,
                "scaled_dot_product_attention",
                self._sdpa_wrapper,
            )
        for name in ("matmul", "bmm", "einsum", "mul", "multiply"):
            self._patch(self.torch, name, self._weighting_wrapper)
        tensor_type = getattr(self.torch, "Tensor", None)
        if tensor_type is not None:
            self._patch(tensor_type, "softmax", self._softmax_wrapper)
            # `weights @ values` dispatches here, not to torch.matmul.
            for name in ("matmul", "__matmul__", "mul", "__mul__", "__rmul__"):
                self._patch(tensor_type, name, self._weighting_wrapper)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self._scores.clear()
        self._score_ids.clear()


def _known_attention_modules(model: Any, torch: Any) -> list[Any]:
    """Return standard attention modules; custom attention is observed at runtime."""
    return [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]


def _snapshot(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return ("tensor", value.detach().cpu().clone())
    if isinstance(value, tuple):
        return ("tuple", tuple(_snapshot(item, torch) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_snapshot(item, torch) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _snapshot(item, torch)) for key, item in value.items()),
        )
    return ("value", repr(value))


def _snapshots_differ(left: Any, right: Any, torch: Any) -> bool:
    if left[0] != right[0]:
        return True
    if left[0] == "tensor":
        return not torch.equal(left[1], right[1])
    if left[0] in {"tuple", "list", "dict"}:
        if len(left[1]) != len(right[1]):
            return True
        if left[0] == "dict":
            return any(
                lkey != rkey or _snapshots_differ(lvalue, rvalue, torch)
                for (lkey, lvalue), (rkey, rvalue) in zip(left[1], right[1])
            )
        return any(
            _snapshots_differ(lvalue, rvalue, torch)
            for lvalue, rvalue in zip(left[1], right[1])
        )
    return left[1] != right[1]


def _corrupt_module_output(output: Any, torch: Any) -> Any:
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (torch.zeros_like(output[0]), *output[1:])
    return output


def _attention_probe(
    solve: Any,
    model: Any,
    cases: list,
    torch: Any,
) -> dict[str, Any]:
    """Detect executed attention and verify that perturbing it changes forward."""
    if not cases:
        return {
            "detected": False,
            "output_dependent": False,
            "dependence_rate": 0.0,
            "mechanisms": [],
        }

    standard_modules = _known_attention_modules(model, torch)
    executed_standard: set[int] = set()

    def mark_standard(attention_module: Any, _inputs: Any, _output: Any) -> None:
        executed_standard.add(id(attention_module))

    def collect_outputs(monitor: _AttentionOpMonitor, corrupt_standard: bool):
        snapshots = []
        captured = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            captured.append(_snapshot(output, torch))

        module_hooks = []
        top_hook = None
        try:
            for attention_module in standard_modules:
                if corrupt_standard:
                    module_hooks.append(
                        attention_module.register_forward_hook(
                            lambda _module, _inputs, output: _corrupt_module_output(
                                output, torch
                            )
                        )
                    )
                else:
                    module_hooks.append(
                        attention_module.register_forward_hook(mark_standard)
                    )
            # Register this after attention hooks so it captures the perturbed
            # value even when the returned model is itself MultiheadAttention.
            top_hook = model.register_forward_hook(capture)
            with monitor, torch.no_grad():
                for case in cases:
                    before = len(captured)
                    solve(model, *(int(operand) for operand in case))
                    snapshots.append(tuple(captured[before:]))
        except Exception:
            return None
        finally:
            if top_hook is not None:
                top_hook.remove()
            for hook in module_hooks:
                hook.remove()
        return snapshots

    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        clean_monitor = _AttentionOpMonitor(torch)
        clean = collect_outputs(clean_monitor, corrupt_standard=False)
        mechanisms = set(clean_monitor.mechanisms)
        if executed_standard:
            mechanisms.add("nn.MultiheadAttention")
        detected = clean is not None and bool(mechanisms)
        if not detected:
            return {
                "detected": False,
                "output_dependent": False,
                "dependence_rate": 0.0,
                "mechanisms": sorted(mechanisms),
            }

        perturbed_monitor = _AttentionOpMonitor(torch, perturb=True)
        perturbed = collect_outputs(perturbed_monitor, corrupt_standard=True)
        if perturbed is None or len(clean) != len(perturbed):
            changed = 0
        else:
            changed = sum(
                len(before) != len(after)
                or any(
                    _snapshots_differ(left, right, torch)
                    for left, right in zip(before, after)
                )
                for before, after in zip(clean, perturbed)
            )
        rate = changed / len(cases)
        return {
            "detected": True,
            "output_dependent": changed > 0,
            "dependence_rate": rate,
            "mechanisms": sorted(mechanisms),
        }
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    submission_path = Path(payload["submission"])
    # Families frozen before the operation field existed carry no entrypoint and
    # are all addition.
    entrypoint = str(payload.get("entrypoint") or "add")
    with contextlib.redirect_stdout(sys.stderr):
        module = _load(submission_path)
        if not hasattr(module, "build_model") or not hasattr(module, entrypoint):
            raise ValueError(
                f"submission must define build_model() and {entrypoint}()"
            )
        solve = getattr(module, entrypoint)
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
            for case in payload["cases"]:
                before = call_counter[0]
                result = solve(model, *(int(operand) for operand in case))
                outputs.append(result if isinstance(result, int) else repr(result))
                model_calls.append(call_counter[0] - before)
        finally:
            hook.remove()

        probe_cases = payload.get("probe_cases", [])
        dependence_rate = _output_dependence(solve, model, probe_cases)
        attention = _attention_probe(solve, model, probe_cases, torch)

    return {
        "outputs": outputs,
        "model_calls": model_calls,
        "dependence_rate": dependence_rate,
        "params": _tensor_count(model, bool(payload.get("count_buffers", False))),
        "has_attention": bool(
            attention["detected"] and attention["output_dependent"]
        ),
        "attention": attention,
        "model_type": model.__class__.__name__,
        "claimed_params": metadata.get("params"),
    }


def _output_dependence(solve: Any, model: Any, probe_cases: list) -> float:
    """Fraction of probe answers that change when the model output is corrupted.

    A genuine solver reads its answer off the model output, so replacing every
    forward result with zeros must move the answer on essentially every probe.
    A submission that calls forward only to satisfy the liveness counter and
    computes the answer in Python leaves its answers unchanged, which this gate
    catches without any task-specific knowledge of the operation.
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
    # The clean pass runs with no hook installed. An earlier version ran a
    # corrupt-hooked warm-up first and discarded its results, which cost a full
    # pass over the probe set and could return 0.0 - reading as "the model is
    # not used" - if that throwaway pass happened to raise.
    clean_answers = []
    for case in probe_cases:
        try:
            clean_answers.append(solve(model, *(int(operand) for operand in case)))
        except Exception:
            return 0.0
    hook = model.register_forward_hook(corrupt)
    try:
        for case, clean in zip(probe_cases, clean_answers):
            try:
                corrupted = solve(model, *(int(operand) for operand in case))
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
