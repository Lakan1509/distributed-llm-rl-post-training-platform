from pathlib import Path

from rl_platform.checkpointing import Checkpoint, CheckpointManager
from rl_platform.grpo import grouped_advantages
from rl_platform.integrations import BackendPlan
from rl_platform.resource_scheduler import ResourceScheduler, Worker


def test_grouped_advantages_are_centered():
    values = grouped_advantages([0.0, 1.0, 2.0])
    assert abs(sum(values)) < 1e-7


def test_checkpoint_roundtrip(tmp_path: Path):
    manager = CheckpointManager(tmp_path, every_steps=10)
    assert manager.should_save(90)
    manager.save(Checkpoint(90, "grpo", "qwen", 0.5, {"ok": True}))
    assert manager.latest().step == 90


def test_resource_scheduler():
    scheduler = ResourceScheduler([Worker("a", 1.0), Worker("b", 2.0)])
    assert scheduler.place(1.5).name == "b"


def test_backend_plan_validation():
    BackendPlan().validate()
