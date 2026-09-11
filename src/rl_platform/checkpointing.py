"""Atomic local checkpoint metadata used by the orchestration layer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    step: int
    algorithm: str
    model: str
    reward_mean: float
    metadata: dict[str, object]


class CheckpointManager:
    def __init__(self, root: str | Path, every_steps: int = 10) -> None:
        self.root = Path(root)
        self.every_steps = every_steps
        self.root.mkdir(parents=True, exist_ok=True)

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.every_steps == 0

    def save(self, checkpoint: Checkpoint) -> Path:
        target = self.root / f"checkpoint-{checkpoint.step}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(checkpoint), indent=2, sort_keys=True))
        tmp.replace(target)
        return target

    def latest(self) -> Checkpoint | None:
        candidates = []
        for path in self.root.glob("checkpoint-*.json"):
            try:
                step = int(path.stem.split("-")[-1])
                candidates.append((step, path))
            except ValueError:
                continue
        if not candidates:
            return None
        _, path = max(candidates)
        return Checkpoint(**json.loads(path.read_text()))
