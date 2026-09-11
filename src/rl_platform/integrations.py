"""Typed backend registry for distributed LLM post-training.

Heavy GPU frameworks stay optional so the control plane remains testable on a
laptop while the same configuration can be used on GPU clusters.
"""
from dataclasses import dataclass
from enum import Enum


class Algorithm(str, Enum):
    GRPO = "grpo"
    PPO = "ppo"
    DPO = "dpo"
    RLHF = "rlhf"


class GenerationBackend(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt-llm"


class DistributedRuntime(str, Enum):
    RAY = "ray"
    MONARCH = "monarch"
    MILES = "miles"


class TrainingBackend(str, Enum):
    PYTORCH_FSDP2 = "pytorch-fsdp2"
    MEGATRON_LM = "megatron-lm"


@dataclass(frozen=True)
class BackendPlan:
    algorithm: Algorithm = Algorithm.GRPO
    generation: GenerationBackend = GenerationBackend.VLLM
    runtime: DistributedRuntime = DistributedRuntime.RAY
    training: TrainingBackend = TrainingBackend.PYTORCH_FSDP2
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    mixed_precision: str = "bf16"

    def validate(self) -> None:
        for name, value in (
            ("tensor_parallel_size", self.tensor_parallel_size),
            ("pipeline_parallel_size", self.pipeline_parallel_size),
            ("expert_parallel_size", self.expert_parallel_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.mixed_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("mixed_precision must be bf16, fp16, or fp32")


CAPABILITIES = {
    "algorithms": [x.value for x in Algorithm],
    "generation": [x.value for x in GenerationBackend],
    "runtimes": [x.value for x in DistributedRuntime],
    "training": [x.value for x in TrainingBackend],
    "parallelism": ["data", "tensor", "pipeline", "expert"],
    "collectives": ["NCCL"],
    "orchestration": ["Docker", "Kubernetes"],
}
