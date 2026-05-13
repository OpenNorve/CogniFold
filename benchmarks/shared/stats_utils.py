"""Statistical utilities for benchmark evaluation.

Provides confidence interval calculations and formatting helpers
for reporting benchmark results with proper uncertainty quantification.
"""

from __future__ import annotations

import math
import random
from typing import Sequence, Tuple


def wilson_ci(
    successes: int, n: int, z: float = 1.96
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal approximation for small samples
    and proportions near 0 or 1.

    Args:
        successes: Number of successes (correct answers).
        n: Total number of trials.
        z: Z-score for desired confidence level (default 1.96 = 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.

    Raises:
        ValueError: If n <= 0 or successes not in [0, n].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(
            f"successes must be in [0, n], got {successes} with n={n}"
        )

    p_hat = successes / n
    z2 = z * z
    denom = 1 + z2 / n

    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1 - p_hat) + z2 / (4 * n)) / n
    )

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (lower, upper)


def bootstrap_ci_mean(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean of continuous metrics.

    Uses the percentile method to compute a (1-alpha) confidence interval.

    Args:
        values: Observed metric values (e.g., per-sample F1 scores).
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Optional random seed for reproducibility.

    Returns:
        (mean, lower, upper) where mean is the observed sample mean
        and lower/upper are the CI bounds.

    Raises:
        ValueError: If values is empty or alpha not in (0, 1).
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = random.Random(seed)
    n = len(values)
    vals = list(values)

    observed_mean = sum(vals) / n

    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo_idx = int(math.floor((alpha / 2) * n_boot))
    hi_idx = int(math.floor((1 - alpha / 2) * n_boot)) - 1

    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))

    return (observed_mean, boot_means[lo_idx], boot_means[hi_idx])


def format_ci(value: float, lower: float, upper: float) -> str:
    r"""Format a value with confidence interval for LaTeX.

    Produces a string like ``0.457 [0.312, 0.602]`` suitable for
    inclusion in LaTeX tables or inline text.

    Args:
        value: Point estimate.
        lower: Lower bound of CI.
        upper: Upper bound of CI.

    Returns:
        Formatted string ``"value [lower, upper]"``.
    """
    return f"{value:.3f} [{lower:.3f}, {upper:.3f}]"
