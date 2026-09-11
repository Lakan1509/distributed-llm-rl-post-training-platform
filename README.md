# Distributed LLM RL Post-Training & Rollout Platform

**PyTorch | VeRL | GRPO | PPO | DPO | RLHF | Ray | Monarch | Miles | Megatron-LM | MoE | vLLM | SGLang | TensorRT-LLM | FSDP2 | NCCL | Kubernetes**

A systems-oriented platform for LLM RL post-training, distributed rollout
generation, reward computation, policy optimization, checkpoint/recovery,
resource-aware scheduling, and production GPU orchestration.

## What is implemented in this repository

- Modular post-training control plane for rollout -> reward -> update workflows
- Algorithm registry for GRPO, PPO, DPO, and RLHF
- Framework-neutral GRPO grouped-advantage and clipped-surrogate utilities
- Generation-backend registry for vLLM, SGLang, and TensorRT-LLM
- Distributed-runtime registry for Ray, Monarch, and Miles
- Training-backend registry for PyTorch FSDP/FSDP2 and Megatron-LM
- Data/tensor/pipeline/expert-parallel topology configuration
- Resource-aware worker placement and worker-recovery primitives
- Atomic configurable checkpointing, including 10-step checkpoint intervals
- NVIDIA/NCCL-oriented Kubernetes rollout-worker and trainer manifests
- Qwen2.5-1.5B-Instruct + GSM8K GRPO reproduction configuration
- Docker, tests, and GitHub Actions CI

## Architecture

```text
Prompts
  |
  v
Rollout Coordinator --> vLLM / SGLang / TensorRT-LLM workers
  |                              |
  |                              v
  +------------------------> Reward functions / reward model
                                 |
                                 v
                         GRPO / PPO / DPO / RLHF
                                 |
                     +-----------+------------+
                     |                        |
                 FSDP/FSDP2              Megatron-LM
                     |                        |
                     +---- NCCL / GPUs -------+
                                 |
                         Checkpoints + Metrics
                                 |
                       Docker / Kubernetes

Runtime orchestration: Ray / Monarch / Miles
Parallelism: data / tensor / pipeline / expert (MoE)
```

## Reproducible GRPO experiment definition

`configs/qwen25_gsm8k_grpo.yaml` defines the intended experiment:

- Qwen2.5-1.5B-Instruct
- 1,000 GSM8K training examples
- 200 held-out evaluation examples
- 90 GRPO steps
- LoRA + verifiable correctness/format rewards
- checkpoint every 10 steps
- A100 80GB target hardware

### Benchmark integrity

The repository **does not claim unmeasured GPU results**. Values such as a
`19.0% -> 22.0% GSM8K` improvement or recovery from `checkpoint-90` become
achieved results only after the GPU run is reproduced and raw logs/results are
committed. See `docs/BENCHMARK_PROTOCOL.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
python -m rl_platform.cli --prompts 32 --workers 4 --rollouts-per-prompt 2
```

## Production extension path

For a GPU environment, wire the typed integration interfaces to VeRL/TRL,
PyTorch distributed, Ray actors, vLLM/SGLang/TensorRT-LLM servers, and
Megatron-LM. Kubernetes manifests already declare NVIDIA GPU resources and NCCL
runtime configuration.

## Repository map

```text
src/rl_platform/
  integrations.py       # algorithms/backends/runtimes
  grpo.py               # GRPO math
  checkpointing.py      # persistent checkpoint/recovery metadata
  resource_scheduler.py # resource-aware worker placement
  rollout.py             # rollout orchestration
  reward.py              # reward abstraction
  trainer.py             # training control plane
configs/
  qwen25_gsm8k_grpo.yaml
k8s/
  rollout-worker.yaml
  trainer-job.yaml
docs/
  TECHNOLOGY_MATRIX.md
  BENCHMARK_PROTOCOL.md
```

## License

MIT
