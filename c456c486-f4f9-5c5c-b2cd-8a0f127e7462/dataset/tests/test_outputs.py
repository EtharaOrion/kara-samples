import json
import math
import os

SCORE = os.environ.get("SCORE_JSON", "/logs/verifier/score.json")
REPORT = os.environ.get("REPORT_JSON", "/logs/verifier/report.json")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def test_r1():
    score = _load(SCORE)["score"]
    assert isinstance(score, (int, float)) and 0.0 <= score <= 1.0


def test_r2():
    for key, value in _load(SCORE).items():
        assert isinstance(value, (int, float)) and math.isfinite(value), key


def test_r3():
    data = _load(SCORE)
    if data["qualified"] == 1.0:
        assert data["accuracy"] >= 0.99 and data["edge_accuracy"] == 1.0


def test_r4():
    data = _load(SCORE)
    if data["score"] == 0.0:
        report = _load(REPORT)
        attributed = (
            bool(report.get("error"))
            or bool(report.get("findings"))
            or not report.get("structurally_valid", True)
            or data.get("accuracy", 0.0) < 0.99
            or data.get("edge_accuracy", 0.0) < 1.0
            or data.get("model_use_rate", 0.0) < 1.0
        )
        assert attributed, "zero score with every gate green is an unattributed zero"


if __name__ == "__main__":
    import sys

    failures = []
    for name, case in sorted(globals().items()):
        if not name.startswith("test_") or not callable(case):
            continue
        try:
            case()
        except AssertionError as exc:
            failures.append("{}: {}".format(name, exc or "assertion failed"))
        except Exception as exc:  # noqa: BLE001
            failures.append("{}: {}: {}".format(name, type(exc).__name__, exc))
    for line in failures:
        print("FAIL " + line, file=sys.stderr)
    print("compiled output tests: {} passed, {} failed".format(
        sum(1 for n in globals() if n.startswith("test_")) - len(failures),
        len(failures),
    ))
    sys.exit(1 if failures else 0)
