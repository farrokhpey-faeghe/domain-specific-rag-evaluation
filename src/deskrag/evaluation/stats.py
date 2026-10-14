"""Exact paired binary inference without scipy."""

import math


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided binomial McNemar p; b=pass→fail, c=fail→pass."""
    if type(b) is not int or type(c) is not int or min(b, c) < 0:
        raise ValueError("discordant counts must be nonnegative integers")
    n = b + c
    if not n:
        return 1.0
    return min(1.0, (2 * sum(math.comb(n, k) for k in range(min(b, c) + 1))) / (1 << n))
