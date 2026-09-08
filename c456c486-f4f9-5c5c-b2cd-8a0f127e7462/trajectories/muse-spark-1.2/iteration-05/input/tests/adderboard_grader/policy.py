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


def inspect_source(path: Path, max_bytes: int = 1_000_000) -> list[str]:
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
    add_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "add"
    ]
    if len(add_functions) != 1:
        findings.append("submission must define exactly one add function")

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
                and node.func.attr in {"add", "iadd"}
            ):
                findings.append("direct operator.add shortcut")

    if add_functions:
        add_function = add_functions[0]
        # The operands are the second and third positional parameters, whatever
        # the submission names them. Hardcoding "a" and "b" let a submission
        # evade every operand check by renaming its parameters, so derive the
        # real names from the signature.
        params = [arg.arg for arg in add_function.args.args]
        operands = set(params[1:3]) if len(params) >= 3 else {"a", "b"}
        for node in ast.walk(add_function):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                if operands.issubset(_names(node)):
                    findings.append("direct addition of add() operands")
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                if operands.intersection(_names(node)):
                    findings.append("augmented addition on add() operands")
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
                if target in {"sum", "fsum", "reduce", "add", "np", "tensor"}:
                    if operands.issubset(_direct_names(node)):
                        findings.append("operands reduced to a sum through a builtin call")

    return sorted(set(findings))

