from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str


@dataclass(frozen=True)
class Rollout:
    prompt_id: str
    response: str
    latency_ms: float
    worker_id: int


@dataclass(frozen=True)
class ScoredRollout:
    rollout: Rollout
    reward: float
