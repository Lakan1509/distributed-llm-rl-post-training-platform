from __future__ import annotations

import argparse

from .models import Prompt
from .rollout import RolloutCoordinator
from .trainer import PostTrainingOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollouts-per-prompt", type=int, default=2)
    args = parser.parse_args()

    prompts = [Prompt(str(i), f"Explain AI concept number {i}") for i in range(args.prompts)]
    trainer = PostTrainingOrchestrator(
        RolloutCoordinator(args.workers, args.rollouts_per_prompt)
    )
    result = trainer.train_step(prompts)
    m = result.metrics
    print(f"prompts={args.prompts} rollouts={m.rollouts} workers={args.workers}")
    print(f"mean_reward={m.mean_reward:.3f} p95_latency_ms={m.p95_latency_ms:.3f}")
    print(f"policy_update=simulated step={result.step}")


if __name__ == "__main__":
    main()
