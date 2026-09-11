from __future__ import annotations

import concurrent.futures
import hashlib
import time
from collections.abc import Iterable

from .models import Prompt, Rollout


class SimulatedRolloutWorker:
    """Laptop-friendly stand-in for a real LLM generation worker."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id

    def generate(self, prompt: Prompt, sample_index: int) -> Rollout:
        start = time.perf_counter()
        digest = hashlib.sha256(f"{prompt.prompt_id}:{sample_index}".encode()).hexdigest()[:12]
        response = f"candidate[{digest}] for: {prompt.text[:80]}"
        latency_ms = (time.perf_counter() - start) * 1000
        return Rollout(prompt.prompt_id, response, latency_ms, self.worker_id)


class RolloutCoordinator:
    def __init__(self, workers: int = 4, rollouts_per_prompt: int = 2) -> None:
        if workers < 1 or rollouts_per_prompt < 1:
            raise ValueError("workers and rollouts_per_prompt must be >= 1")
        self.workers = workers
        self.rollouts_per_prompt = rollouts_per_prompt

    def run(self, prompts: Iterable[Prompt]) -> list[Rollout]:
        jobs: list[tuple[int, Prompt, int]] = []
        for prompt in prompts:
            for sample_index in range(self.rollouts_per_prompt):
                jobs.append((len(jobs) % self.workers, prompt, sample_index))

        worker_pool = [SimulatedRolloutWorker(i) for i in range(self.workers)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(worker_pool[worker_id].generate, prompt, sample_index)
                for worker_id, prompt, sample_index in jobs
            ]
            return [future.result() for future in futures]
