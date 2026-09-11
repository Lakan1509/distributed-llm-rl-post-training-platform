from rl_platform.models import Prompt
from rl_platform.rollout import RolloutCoordinator
from rl_platform.trainer import PostTrainingOrchestrator


def test_rollout_count():
    prompts = [Prompt("a", "hello"), Prompt("b", "world")]
    rollouts = RolloutCoordinator(workers=2, rollouts_per_prompt=3).run(prompts)
    assert len(rollouts) == 6


def test_train_step():
    trainer = PostTrainingOrchestrator(RolloutCoordinator(workers=2, rollouts_per_prompt=2))
    result = trainer.train_step([Prompt("a", "hello")])
    assert result.step == 1
    assert result.metrics.rollouts == 2
    assert 0.0 <= result.metrics.mean_reward <= 1.0
