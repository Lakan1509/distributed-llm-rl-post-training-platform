from __future__ import annotations

import hashlib

from .models import Rollout, ScoredRollout


class DeterministicRewardScorer:
    """Deterministic reward stub that can be replaced by a learned reward model."""

    def score(self, rollout: Rollout) -> ScoredRollout:
        raw = hashlib.sha256(rollout.response.encode()).digest()[0]
        reward = round(raw / 255.0, 4)
        return ScoredRollout(rollout=rollout, reward=reward)
