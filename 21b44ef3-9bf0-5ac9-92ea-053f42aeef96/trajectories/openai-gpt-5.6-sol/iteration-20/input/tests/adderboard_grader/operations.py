"""What each arithmetic family computes, and the domain it is graded over.

One module owns the operation-specific facts, so the grader, the generator, and
the shipped baseline cannot disagree about them. Everything else in this package
stays deliberately operation-blind: the attention probe, the output-dependence
gate, the parameter recount, and the score curve all work the same way whatever
the family computes, and none of them should ever need to know.

Subtraction is graded over *ordered* operands. The admitted domain is
`{(a, b) : 0 <= b <= a <= maximum}`, so every result is a non-negative integer
no wider than the operands and no sign token enters the vocabulary.

That choice is what keeps the borrow chain an exact structural twin of
addition's carry chain. Over uniform digits a column generates a borrow when
`a_i < b_i` (probability 0.45) and propagates one when `a_i == b_i`
(probability 0.1); addition generates a carry when `s >= 10` (probability 0.45)
and propagates one when `s == 9` (probability 0.1). The two families therefore
see identically distributed chain lengths, chained-precision survival means the
same thing in both, and the difficulty ladder stays comparable across them.

The graded set is drawn uniformly over that ordered domain, which is exactly the
swap of two independent uniforms. Drawing `b` first and then `a` from
`[b, maximum]` would *not* be uniform over the domain: it over-weights large `b`
and so under-represents the long borrow chains the task exists to test.
"""

from __future__ import annotations


class Operation:
    """One arithmetic family: its ground truth, its domain, and its wording."""

    # Family name, and the prefix every variant slug in the family must carry.
    name: str
    # Number of graded operands. Two for the classic families; the batch-001
    # three-operand families carry three. Every case tuple, probe tuple, and
    # entrypoint signature has exactly this many operands.
    arity: int = 2
    # Name of the graded function in submission.py.
    entrypoint: str
    # Wording used in generated agent-visible text.
    verb: str
    noun: str
    gerund: str
    symbol: str
    # Noun for the baseline model, e.g. "KARA baseline adder".
    agent_noun: str
    # Class name of the shipped baseline model.
    model_class: str
    # Name of the per-column carry-equivalent, used in generated prose.
    chain_noun: str

    def result_digits(self, digits: int) -> int:
        """Width of the result, in decimal digits, for a `digits`-wide operand."""
        raise NotImplementedError

    def apply(self, *operands: int) -> int:
        """The ground truth. This is the only definition of correctness."""
        raise NotImplementedError

    def order(self, *operands: int) -> tuple[int, ...]:
        """Map an arbitrary operand tuple into the family's admitted domain."""
        raise NotImplementedError

    def draw(self, rng, maximum: int) -> tuple[int, ...]:
        """One graded case, drawn the way the grader draws its random set.

        The default draws `arity` independent uniforms over `[0, maximum]` and
        maps them through `order`. The argument generator is consumed left to
        right, so for the classic two-operand families this makes exactly the
        same `rng.randint` call sequence the grader made before `draw` existed,
        and every frozen seed reproduces its original case list byte for byte.
        """
        return self.order(*(rng.randint(0, maximum) for _ in range(self.arity)))

    def edge_cases(self, maximum: int) -> list[tuple[int, ...]]:
        """Fixed cases the score gates at one hundred percent."""
        raise NotImplementedError

    def probe_pair(self, rng, maximum: int) -> tuple[int, ...]:
        """One operand tuple for the output-dependence gate.

        Arity-length despite the historical name. Never degenerate: the answer
        must be something a model corrupted to zeros cannot also produce, or an
        honest submission fails the gate.
        """
        raise NotImplementedError

    def domain_line(self, maximum: int) -> str:
        """The interface line in instruction.md that states the domain."""
        raise NotImplementedError


class _Addition(Operation):
    name = "add"
    entrypoint = "add"
    verb = "add"
    noun = "sum"
    gerund = "addition"
    symbol = "+"
    agent_noun = "adder"
    model_class = "AdderTransformer"
    chain_noun = "carry"

    def result_digits(self, digits: int) -> int:
        return digits + 1

    def apply(self, a: int, b: int) -> int:
        return a + b

    def order(self, a: int, b: int) -> tuple[int, int]:
        # Addition admits the whole square, so ordering is the identity. It must
        # stay the identity: reordering here would change the seeded case
        # sequence of every existing addition task.
        return a, b

    def edge_cases(self, maximum: int) -> list[tuple[int, int]]:
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

    def probe_pair(self, rng, maximum: int) -> tuple[int, int]:
        return rng.randint(1, maximum), rng.randint(1, maximum)

    def domain_line(self, maximum: int) -> str:
        return (
            f"- `add(model, a: int, b: int) -> int` returns the exact sum for "
            f"operands in `[0, {maximum:,}]`."
        )


class _Subtraction(Operation):
    name = "subtract"
    entrypoint = "subtract"
    verb = "subtract"
    noun = "difference"
    gerund = "subtraction"
    symbol = "-"
    agent_noun = "subtractor"
    model_class = "SubtractorTransformer"
    chain_noun = "borrow"

    def result_digits(self, digits: int) -> int:
        # Ordered operands, so the difference never exceeds the minuend and
        # never needs a sign slot. One digit narrower than the addition family
        # at the same width.
        return digits

    def apply(self, a: int, b: int) -> int:
        return a - b

    def order(self, a: int, b: int) -> tuple[int, int]:
        return (a, b) if a >= b else (b, a)

    def edge_cases(self, maximum: int) -> list[tuple[int, int]]:
        # The smallest number of the graded width, i.e. 10 ** (digits - 1). Its
        # single leading one over a run of zeros is what forces the longest
        # borrow chain, and it is the subtraction analogue of addition's
        # all-nines carry chain.
        power = (maximum + 1) // 10
        return [
            (0, 0),
            (1, 0),
            (1, 1),
            (maximum, 0),
            (maximum, maximum),
            (maximum, 1),
            # Longest borrow chain: every column below the leading digit
            # propagates.
            (power, 1),
            # Borrow through every column at once, result exactly one.
            (power, power - 1),
            # Result one with no borrow anywhere, the contrast case for the
            # pair above.
            (maximum, maximum - 1),
            # Alternating borrows: 9090...90 minus 1111...11 borrows on every
            # other column, which no single fixed offset can satisfy.
            (maximum // 11 * 10, maximum // 9),
        ]

    def probe_pair(self, rng, maximum: int) -> tuple[int, int]:
        # Strictly greater, so the difference is never zero. A zero answer is
        # what a model corrupted to zeros may also decode, which would read as
        # "the answer does not depend on the model" for an honest submission.
        b = rng.randint(1, maximum - 1)
        return rng.randint(b + 1, maximum), b

    def domain_line(self, maximum: int) -> str:
        return (
            f"- `subtract(model, a: int, b: int) -> int` returns the exact "
            f"difference `a - b` for operands in `[0, {maximum:,}]`. The grader "
            f"always passes `a >= b`, so the result is never negative and never "
            f"wider than the operands."
        )


def _band_floor(maximum: int) -> int:
    """Smallest value of the graded width: 10 ** (digits - 1).

    The batch-001 families grade full-width operands only, per the approved
    contract's operand-range resolution: every operand carries exactly the
    task's digit width, no leading zeros. The classic families above grade the
    whole `[0, maximum]` square and must stay that way, because changing their
    domain would silently change the seeded case sequence of every existing
    task."""
    return (maximum + 1) // 10


class _BandAddition2(Operation):
    """Two-operand addition over full-width operands only."""

    name = "add2"
    arity = 2
    entrypoint = "add"
    verb = "add"
    noun = "sum"
    gerund = "addition"
    symbol = "+"
    agent_noun = "adder"
    model_class = "AdderTransformer"
    chain_noun = "carry"

    def result_digits(self, digits: int) -> int:
        return digits + 1

    def apply(self, *operands: int) -> int:
        a, b = operands
        return a + b

    def order(self, *operands: int) -> tuple[int, ...]:
        return tuple(operands)

    def draw(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        return rng.randint(floor, maximum), rng.randint(floor, maximum)

    def edge_cases(self, maximum: int) -> list[tuple[int, ...]]:
        floor = _band_floor(maximum)
        mid = (floor + maximum) // 2
        return [
            (floor, floor),
            (floor, floor + 1),
            # All-nines against all-nines: a carry out of every column.
            (maximum, maximum),
            (maximum, floor),
            (floor, maximum),
            # All-nines plus one-zeros-one: the carry cascades through every
            # column at once, the band analogue of the classic (max, 1) chain.
            (maximum, floor + 1),
            (floor + 1, maximum),
            # Identical operands, the doubling case.
            (mid, mid),
            # Alternating columns: 9090...90 plus 1111...11 carries on every
            # other column, which no single fixed offset satisfies.
            (maximum // 11 * 10, maximum // 9),
            (maximum - 1, floor),
        ]

    def probe_pair(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        return rng.randint(floor, maximum), rng.randint(floor, maximum)

    def domain_line(self, maximum: int) -> str:
        floor = _band_floor(maximum)
        return (
            f"- `add(model, a: int, b: int) -> int` returns the exact sum for "
            f"operands in `[{floor:,}, {maximum:,}]`. Every graded operand "
            f"carries the full digit width, never fewer digits."
        )


class _BandSubtraction2(Operation):
    """Two-operand ordered subtraction over full-width operands only."""

    name = "sub2"
    arity = 2
    entrypoint = "subtract"
    verb = "subtract"
    noun = "difference"
    gerund = "subtraction"
    symbol = "-"
    agent_noun = "subtractor"
    model_class = "SubtractorTransformer"
    chain_noun = "borrow"

    def result_digits(self, digits: int) -> int:
        return digits

    def apply(self, *operands: int) -> int:
        a, b = operands
        return a - b

    def order(self, *operands: int) -> tuple[int, ...]:
        a, b = operands
        return (a, b) if a >= b else (b, a)

    def draw(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        return self.order(
            rng.randint(floor, maximum), rng.randint(floor, maximum)
        )

    def edge_cases(self, maximum: int) -> list[tuple[int, ...]]:
        floor = _band_floor(maximum)
        mid = (floor + maximum) // 2
        return [
            (floor, floor),
            (maximum, maximum),
            (maximum, floor),
            (floor + 1, floor),
            (maximum, maximum - 1),
            # Two-zeros minus one-zeros-one: the borrow cascades through every
            # column below the lead, the band analogue of (power, 1).
            (2 * floor, floor + 1),
            # Borrow through every column at once, result exactly one.
            (2 * floor, 2 * floor - 1),
            # Alternating borrows: 9090...90 minus 1111...11 borrows on every
            # other column.
            (maximum // 11 * 10, maximum // 9),
            (maximum, maximum // 9),
            (mid, mid),
        ]

    def probe_pair(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        b = rng.randint(floor, maximum - 1)
        return rng.randint(b + 1, maximum), b

    def domain_line(self, maximum: int) -> str:
        floor = _band_floor(maximum)
        return (
            f"- `subtract(model, a: int, b: int) -> int` returns the exact "
            f"difference `a - b` for operands in `[{floor:,}, {maximum:,}]`. "
            f"The grader always passes `a >= b`, so the result is never "
            f"negative, and every graded operand carries the full digit width."
        )


class _Addition3(Operation):
    """Three-operand addition over full-width operands only.

    Per-column sums reach 9 + 9 + 9 plus an incoming carry of up to two, so the
    carry alphabet is {0, 1, 2} rather than addition's {0, 1}. That widened
    chain state is the whole point of the family: no public two-operand
    construction hardcodes it.
    """

    name = "add3"
    arity = 3
    entrypoint = "add3"
    verb = "add"
    noun = "sum"
    gerund = "three-operand addition"
    symbol = "+"
    agent_noun = "adder"
    model_class = "Adder3Transformer"
    chain_noun = "carry"

    def result_digits(self, digits: int) -> int:
        # Three addends of `digits` width sum below 3 * 10**digits, which always
        # fits in digits + 1 columns.
        return digits + 1

    def apply(self, *operands: int) -> int:
        a, b, c = operands
        return a + b + c

    def order(self, *operands: int) -> tuple[int, ...]:
        return tuple(operands)

    def draw(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        return (
            rng.randint(floor, maximum),
            rng.randint(floor, maximum),
            rng.randint(floor, maximum),
        )

    def edge_cases(self, maximum: int) -> list[tuple[int, ...]]:
        floor = _band_floor(maximum)
        mid = (floor + maximum) // 2
        return [
            (floor, floor, floor),
            (floor + 1, floor, floor),
            (maximum, floor, floor),
            (maximum, maximum, floor),
            # Sustained carry-two chain: every column sums 9 + 9 + 9 plus an
            # incoming two, which only the three-operand family can produce.
            (maximum, maximum, maximum),
            # Full-length cascade seeded at the low column.
            (maximum, maximum, floor + 1),
            # Alternating columns under a doubled addend load.
            (maximum // 11 * 10, maximum // 9, maximum // 9),
            # Identical operands, the tripling case.
            (mid, mid, mid),
            (maximum, floor, maximum),
            (maximum - 1, maximum - 1, maximum - 1),
        ]

    def probe_pair(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        return (
            rng.randint(floor, maximum),
            rng.randint(floor, maximum),
            rng.randint(floor, maximum),
        )

    def domain_line(self, maximum: int) -> str:
        floor = _band_floor(maximum)
        return (
            f"- `add3(model, a: int, b: int, c: int) -> int` returns the exact "
            f"sum `a + b + c` for operands in `[{floor:,}, {maximum:,}]`. Every "
            f"graded operand carries the full digit width, never fewer digits."
        )


class _Subtraction3(Operation):
    """Three-operand ordered subtraction over full-width operands only.

    The admitted domain is `{(a, b, c) : b + c <= a}` with every operand at the
    full graded width, so the result is never negative and never wider than the
    minuend. Column borrows can reach two, the mirror of the three-operand
    carry alphabet.
    """

    name = "sub3"
    arity = 3
    entrypoint = "subtract3"
    verb = "subtract"
    noun = "difference"
    gerund = "three-operand subtraction"
    symbol = "-"
    agent_noun = "subtractor"
    model_class = "Subtractor3Transformer"
    chain_noun = "borrow"

    def result_digits(self, digits: int) -> int:
        return digits

    def apply(self, *operands: int) -> int:
        a, b, c = operands
        return a - b - c

    def order(self, *operands: int) -> tuple[int, ...]:
        """Domain map for authored tuples: largest first, then feasibility.

        Random draws never route through here; `draw` samples the admitted
        domain directly. This exists so an authored edge case that names its
        operands in the wrong order still lands inside the domain, and a tuple
        no ordering can repair raises rather than grading an unreachable case.
        """
        ordered = sorted(operands, reverse=True)
        a, b, c = ordered
        if b + c > a:
            raise ValueError(
                f"no ordering of {operands!r} satisfies b + c <= a"
            )
        return a, b, c

    def draw(self, rng, maximum: int) -> tuple[int, ...]:
        """Uniform over the admitted domain by rejection.

        Rejection is the only draw that is actually uniform over the domain:
        any deterministic reordering of three independent uniforms over-weights
        the region where the two subtrahends are small, which under-represents
        exactly the long double-borrow chains the family exists to test. The
        acceptance rate is about one in nine at every width, so the seeded loop
        terminates fast and deterministically."""
        floor = _band_floor(maximum)
        while True:
            a = rng.randint(floor, maximum)
            b = rng.randint(floor, maximum)
            c = rng.randint(floor, maximum)
            if b + c <= a:
                return a, b, c

    def edge_cases(self, maximum: int) -> list[tuple[int, ...]]:
        floor = _band_floor(maximum)
        return [
            # Minimal slack: the result is exactly zero.
            (2 * floor, floor, floor),
            (2 * floor + 1, floor, floor),
            (maximum, floor, floor),
            # b + c equals a exactly, full-width cancellation.
            (maximum, floor, maximum - floor),
            # Near-total cancellation, result exactly one.
            (maximum, maximum // 2, maximum // 2),
            # Cascading double borrow through every column.
            (3 * floor, floor + 1, floor + 2),
            (maximum, 2 * floor, floor),
            # Result exactly two under a doubled subtrahend load.
            (4 * floor, 2 * floor - 1, 2 * floor - 1),
            (maximum, floor, floor + 1),
            (maximum - 1, floor, floor),
        ]

    def probe_pair(self, rng, maximum: int) -> tuple[int, ...]:
        floor = _band_floor(maximum)
        while True:
            a = rng.randint(floor, maximum)
            b = rng.randint(floor, maximum)
            c = rng.randint(floor, maximum)
            # Strictly greater, so the difference is never zero and a
            # corrupted-to-zeros model output genuinely moves the answer.
            if b + c < a:
                return a, b, c

    def domain_line(self, maximum: int) -> str:
        floor = _band_floor(maximum)
        return (
            f"- `subtract3(model, a: int, b: int, c: int) -> int` returns the "
            f"exact difference `a - b - c` for operands in "
            f"`[{floor:,}, {maximum:,}]`. The grader always passes operands "
            f"with `b + c <= a`, so the result is never negative, and every "
            f"graded operand carries the full digit width."
        )


ADD = _Addition()
SUBTRACT = _Subtraction()
ADD2 = _BandAddition2()
SUB2 = _BandSubtraction2()
ADD3 = _Addition3()
SUB3 = _Subtraction3()

OPERATIONS: dict[str, Operation] = {
    operation.name: operation
    for operation in (ADD, SUBTRACT, ADD2, SUB2, ADD3, SUB3)
}

# Families frozen before the operation field existed carry no `operation` key in
# their spec.json, and they are all addition.
DEFAULT_OPERATION = ADD.name


def resolve(name: str | None) -> Operation:
    operation = OPERATIONS.get(name or DEFAULT_OPERATION)
    if operation is None:
        raise ValueError(
            f"unknown operation {name!r}; known: {sorted(OPERATIONS)}"
        )
    return operation
