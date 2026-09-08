"""Bounded reward calculation for AdderBoard tasks.

The reward is one float in the closed interval [0, 1], continuous and monotone in
every quantity it scores, so an agent refining its own submission always sees a
gradient in the direction it should move.

    reward = base(params) * accuracy_factor(accuracy) * edge_factor(edge_accuracy)

    base             0.5 at the baseline parameter count, rising to 1.0 at the
                     declared parameter floor. Logarithmic, so every halving of
                     the model is worth the same amount all the way down and
                     there is no size at which shrinking stops paying.
    accuracy_factor  1.0 at or above the qualification threshold, decaying to 0
                     at zero accuracy.
    edge_factor      1.0 with every fixed edge case exact, decaying steeply but
                     continuously as edge cases are missed.

Both factors are exactly 1.0 for a qualifying submission, so a qualifying
submission still scores in [0.5, 1.0], and full reward keeps one exact declared
value: the parameter floor, at or above threshold, with every edge case exact.

The converse no longer holds. A near-threshold submission carrying a very small
model can now score above 0.5 without qualifying. That overlap is the price of
continuity: a strictly banded reward must jump at the band boundary, and that
jump is what stalls a refinement loop, because every exploratory step across the
threshold costs more than the step that motivated it. Consumers that need the
hard predicate read the separate `qualified` field, which is unchanged.

Integrity signals stay hard gates. Structural validity, a positive parameter
count, and full model-use coverage are cheat detectors rather than quality
measures, and partial credit for them would be partial credit for cheating.
"""

from __future__ import annotations

import math


# Reward of a qualifying submission whose model is no smaller than the baseline.
QUALIFIED_FLOOR = 0.5
# Convexity of the sub-threshold accuracy decay. Above 1.0 the last few points of
# accuracy count for more than the first few, which matches the difficulty curve
# of exact-match arithmetic without flattening early progress to nothing.
#
# At 3.0 reaching the threshold from 95% is worth about 1.8 model halvings, so
# the qualifying constraint clearly dominates the size objective, and the band
# where an unqualified submission can outscore a qualified one is confined to
# accuracies above roughly 79%. Raising it further tightens both at the cost of
# the early gradient: by 6.0 a model at half accuracy retains under 2% of its
# score, which is the flat region the previous reward function suffered from.
# The right value depends on where attempts actually land on the accuracy axis,
# which the first pilot measures.
ACCURACY_SHARPNESS = 3.0
# The fixed edge cases are the only full-length carry chains in the graded set,
# so missing one is a real failure and decays faster than ordinary accuracy.
EDGE_SHARPNESS = 3.0
# Parameter count at which the size score saturates at 1.0.
#
# Eight, not one. One was the true arithmetic minimum and exactly the wrong
# target: every one-parameter submission observed was a disguised sum, because a
# single learned scalar cannot express digit-wise addition and the only way to
# reach it is to compute the answer outside the model. Setting full reward there
# aimed the agent at the known cheat. Eight sits just below the eleven-parameter
# hand-derived reference, so full reward is reachable, demonstrably hard, and
# still leaves headroom for a solution better than the reference.
DEFAULT_PARAMETER_FLOOR = 8

# Size at which the score reaches zero. Above the baseline the model is no
# smaller than the shipped starting implementation, so it has earned nothing on
# the size axis, but a hard zero there erases the difference between an
# eighteen-thousand parameter model and a two-thousand one. Both scored an
# identical 0.5000 across four real solves spanning eleven times in size, and an
# agent shown that correctly concludes shrinking bought nothing.
DEFAULT_WORST_MULTIPLE = 10.0

# Efficiency credited for landing exactly at the baseline. Crossing it is a real
# event, so it is worth a step rather than a smooth ramp: everything worse than
# the starting implementation is compressed below this value, everything better
# is spread above it.
BASELINE_SPLIT = 0.15


def parameter_efficiency(
    params: int,
    baseline: int,
    floor: int,
    worst: int | None = None,
    split: float = BASELINE_SPLIT,
) -> float:
    """Size score in [0, 1], piecewise log, with the baseline as a threshold.

        params >= worst        0
        baseline < p < worst   compressed into [0, split)
        floor < p <= baseline  spread across [split, 1]
        params <= floor        1

    Two regions rather than one, because the baseline means something: it is the
    size of the shipped starting implementation. A model no smaller than it has
    not improved on the starting point, and the score says so by keeping it below
    `split`. But "has not improved" is not the same as "indistinguishable", and
    the previous hard zero above the baseline made an eighteen-thousand parameter
    model score exactly what a two-thousand parameter one did.
    """
    if params <= 0 or floor <= 0 or baseline <= floor:
        return 0.0
    limit = int(worst) if worst else int(baseline * DEFAULT_WORST_MULTIPLE)
    if limit <= baseline:
        limit = int(baseline * DEFAULT_WORST_MULTIPLE)
    if params >= limit:
        return 0.0
    if params <= floor:
        return 1.0
    if params > baseline:
        ratio = split * math.log(limit / params) / math.log(limit / baseline)
    else:
        ratio = split + (1.0 - split) * (
            math.log(baseline / params) / math.log(baseline / floor)
        )
    return max(0.0, min(1.0, ratio))


def accuracy_factor(
    accuracy: float, threshold: float, sharpness: float = ACCURACY_SHARPNESS
) -> float:
    """1.0 at or above the threshold, decaying to 0 at zero accuracy.

    Reaching exactly 1.0 at the threshold is what makes the reward continuous
    there: the sub-threshold shape meets the qualifying value instead of jumping
    to it.
    """
    if threshold <= 0.0:
        return 0.0
    if accuracy >= threshold:
        return 1.0
    return max(0.0, min(1.0, accuracy / threshold)) ** sharpness


def edge_factor(edge_accuracy: float, sharpness: float = EDGE_SHARPNESS) -> float:
    """1.0 when every fixed edge case is exact, decaying steeply below."""
    return max(0.0, min(1.0, edge_accuracy)) ** sharpness


def bounded_reward(
    *,
    structurally_valid: bool,
    accuracy: float,
    edge_accuracy: float,
    model_use_rate: float,
    params: int,
    threshold: float,
    baseline: int,
    floor: int = DEFAULT_PARAMETER_FLOOR,
    worst: int | None = None,
) -> tuple[float, float, bool]:
    efficiency = parameter_efficiency(params, baseline, floor, worst)
    if not structurally_valid or params <= 0 or model_use_rate < 1.0:
        return 0.0, efficiency, False
    qualified = accuracy >= threshold and edge_accuracy >= 1.0
    base = QUALIFIED_FLOOR + (1.0 - QUALIFIED_FLOOR) * efficiency
    reward = (
        base
        * accuracy_factor(accuracy, threshold)
        * edge_factor(edge_accuracy)
    )
    return max(0.0, min(1.0, reward)), efficiency, qualified
