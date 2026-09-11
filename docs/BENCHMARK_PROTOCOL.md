# GPU Reproduction Protocol

This repository contains the architecture and benchmark configuration for the
resume-level experiments. Exact numerical claims must be produced by a real run
and committed with raw logs before they are presented as achieved results.

## Target experiment

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: GSM8K
- Train split sample: 1,000 examples
- Held-out evaluation: 200 examples
- Optimizer: GRPO, 90 steps
- LoRA: enabled
- Checkpoint interval: 10 steps
- Target hardware: NVIDIA A100 80GB
- Rollout backend: vLLM; SGLang and TensorRT-LLM are comparison backends
- Distributed runtime: Ray; Monarch and Miles are runtime integration targets
- Training: PyTorch FSDP/FSDP2; Megatron-LM is the scale-out comparison backend
- Collectives: NCCL

## Required metrics

Record baseline and final GSM8K exact match, reward mean/variance, gradient norm,
entropy, completion length, clipping ratio, rollout latency, step time, GPU
utilization, GPU memory, and checkpoint recovery behavior.

## Reproduction gate

Do **not** convert target values such as `19% -> 22%`, `checkpoint-90`, or A100
throughput into achieved claims until the corresponding raw logs and result JSON
are checked into `results/`.
