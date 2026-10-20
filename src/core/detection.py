"""Domain-agnostic statistical hypothesis testing for taggant detection (binomial tests, p-values)."""

from typing import Any
import math


def binomial_p_value(successes: int, trials: int, null_prob: float = 0.5) -> float:
    """Compute one-sided binomial test p-value: P(X >= successes) under null probability.

    Args:
        successes: Number of successful detections/matches.
        trials: Total number of independent trials.
        null_prob: Success probability under null hypothesis H0 (default: 0.5).

    Returns:
        One-sided p-value.
    """
    if trials <= 0:
        return 1.0
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")

    # Sum probabilities of k successes for k in range(successes, trials + 1)
    p_val = 0.0
    for k in range(successes, trials + 1):
        comb = math.comb(trials, k)
        p_val += comb * (null_prob**k) * ((1.0 - null_prob) ** (trials - k))
    return min(max(p_val, 0.0), 1.0)
