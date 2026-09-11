from __future__ import annotations

from dataclasses import dataclass

from .models import ScoredRollout


@dataclass(frozen=True)
class RunMetrics:
    rollouts: int
    mean_reward: float
    p95_latency_ms: float


def summarize(items: list[ScoredRollout]) -> RunMetrics:
    if not items:
        return RunMetrics(0, 0.0, 0.0)
    rewards = [item.reward for item in items]
    latencies = sorted(item.rollout.latency_ms for item in items)
    p95_index = min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
    return RunMetrics(
        rollouts=len(items),
        mean_reward=sum(rewards) / len(rewards),
        p95_latency_ms=latencies[p95_index],
    )
