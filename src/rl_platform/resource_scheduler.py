"""Resource-aware placement model for rollout/training workers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Worker:
    name: str
    gpu_capacity: float
    gpu_used: float = 0.0
    healthy: bool = True

    @property
    def free_gpu(self) -> float:
        return max(0.0, self.gpu_capacity - self.gpu_used)


class ResourceScheduler:
    def __init__(self, workers: list[Worker]) -> None:
        self.workers = workers

    def place(self, gpu_request: float) -> Worker:
        eligible = [w for w in self.workers if w.healthy and w.free_gpu >= gpu_request]
        if not eligible:
            raise RuntimeError("no healthy worker has enough free GPU capacity")
        worker = max(eligible, key=lambda w: w.free_gpu)
        worker.gpu_used += gpu_request
        return worker

    def recover(self, worker_name: str) -> None:
        for worker in self.workers:
            if worker.name == worker_name:
                worker.healthy = True
                worker.gpu_used = 0.0
                return
        raise KeyError(worker_name)
