"""Small, framework-neutral GRPO math utilities."""
from __future__ import annotations

from math import sqrt
from typing import Iterable


def grouped_advantages(rewards: Iterable[float], eps: float = 1e-8) -> list[float]:
    values = [float(x) for x in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = sqrt(variance + eps)
    return [(x - mean) / std for x in values]


def clipped_surrogate(
    log_ratio: float,
    advantage: float,
    clip_eps: float = 0.2,
) -> float:
    ratio = 2.718281828459045 ** log_ratio
    clipped = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
    return min(ratio * advantage, clipped * advantage)
