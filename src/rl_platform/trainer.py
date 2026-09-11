from __future__ import annotations

from dataclasses import dataclass

from .metrics import RunMetrics, summarize
from .models import Prompt
from .reward import DeterministicRewardScorer
from .rollout import RolloutCoordinator


@dataclass(frozen=True)
class TrainingResult:
    step: int
    metrics: RunMetrics


class PostTrainingOrchestrator:
    """Coordinates rollout -> reward -> simulated policy update."""

    def __init__(self, coordinator: RolloutCoordinator) -> None:
        self.coordinator = coordinator
        self.scorer = DeterministicRewardScorer()
        self.step = 0

    def train_step(self, prompts: list[Prompt]) -> TrainingResult:
        rollouts = self.coordinator.run(prompts)
        scored = [self.scorer.score(item) for item in rollouts]
        self.step += 1
        return TrainingResult(step=self.step, metrics=summarize(scored))
