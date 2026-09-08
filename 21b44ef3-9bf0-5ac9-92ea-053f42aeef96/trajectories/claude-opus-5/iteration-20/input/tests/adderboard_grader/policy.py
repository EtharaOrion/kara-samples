"""Static negative controls for obvious verifier and arithmetic shortcuts."""

from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {
    "collections",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "numpy",
    "torch",
    "typing",
}
# Attributes of the `operator` module that hand back a bare arithmetic
# primitive. Both families are screened against the whole set rather than
# against their own operator alone: `a - (-b)` is a sum and `a + (-b)` is a
# difference, so screening only the family's own sign is not a screen at all.
FORBIDDEN_OPERATOR_ATTRS = {"add", "iadd", "sub", "isub"}
# Callables that reduce two operands to one value. `np.subtract(a, b)` routed
# around this set entirely until it was added.
REDUCER_NAMES = {
    "sum",
    "fsum",
    "reduce",
    "add",
    "sub",
    "subtract",
    "difference",
    "np",
    "tensor",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
}


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _direct_names(call: ast.Call) -> set[str]:
    """Names a call receives in its own arguments, not through a nested call.

    Descending into nested calls would make every expression that merely mentions
    the operands somewhere below a reducer look like a laundered sum. Nested
    calls are opaque here; arithmetic inside them is caught by the BinOp rule.
    """
    found: set[str] = set()
    for argument in list(call.args) + [kw.value for kw in call.keywords]:
        stack = [argument]
        while stack:
            current = stack.pop()
            if isinstance(current, ast.Call):
                continue
            if isinstance(current, ast.Name):
                found.add(current.id)
            stack.extend(ast.iter_child_nodes(current))
    return found


def inspect_source(
    path: Path, entrypoint: str = "add", arity: int = 2, max_bytes: int = 1_000_000
) -> list[str]:
    if not path.is_file():
        return ["submission.py is missing"]
    if path.stat().st_size > max_bytes:
        return [f"submission.py exceeds {max_bytes} bytes"]
    source = path.read_text(encoding="utf-8", errors="strict")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"submission.py does not parse: {exc.msg}"]

    findings: list[str] = []
    graded_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entrypoint
    ]
    if len(graded_functions) != 1:
        findings.append(
            f"submission must define exactly one {entrypoint} function"
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                root = module.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    findings.append(f"forbidden import: {module}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                findings.append(f"forbidden call: {node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "operator"
                and node.func.attr in FORBIDDEN_OPERATOR_ATTRS
            ):
                findings.append(f"direct operator.{node.func.attr} shortcut")

    if graded_functions:
        graded_function = graded_functions[0]
        # The operands are the positional parameters after the model, whatever
        # the submission names them and however many the family grades.
        # Hardcoding "a" and "b" let a submission evade every operand check by
        # renaming its parameters, so derive the real names from the signature.
        params = [arg.arg for arg in graded_function.args.args]
        operands = (
            set(params[1 : 1 + arity])
            if len(params) >= 1 + arity
            else set("abcdefgh"[:arity])
        )
        for node in ast.walk(graded_function):
            # Any two graded operands meeting under a single addition or
            # subtraction is direct arithmetic. Requiring the full operand set
            # was right at arity two, where two is the full set, but at arity
            # three it would wave through `a + b` computed beside the model.
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                if len(operands & _names(node)) >= 2:
                    findings.append(
                        f"direct arithmetic on the {entrypoint}() operands"
                    )
            if isinstance(node, ast.AugAssign) and isinstance(
                node.op, (ast.Add, ast.Sub)
            ):
                if operands.intersection(_names(node)):
                    findings.append(
                        f"augmented arithmetic on the {entrypoint}() operands"
                    )
            # Laundering the sum through a reducer or reduction builtin over the
            # operands routes around the direct-BinOp check, so flag a call that
            # passes both operands to a reducer.
            #
            # Only *directly* passed operands count. `torch.tensor([encode(a, b)])`
            # is the ordinary way to build model input from the operands and must
            # not be flagged, whereas `sum([a, b])` and `np.add(a, b)` must be.
            # The distinction is whether the operands reach the reducer itself or
            # merely appear somewhere inside a nested call, and any real
            # arithmetic on them is already caught by the BinOp check above.
            if isinstance(node, ast.Call):
                target = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                if target in REDUCER_NAMES:
                    if len(operands & _direct_names(node)) >= 2:
                        findings.append(
                            "operands reduced to one value through a builtin call"
                        )

    return sorted(set(findings))

