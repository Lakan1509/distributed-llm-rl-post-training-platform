# Technology Matrix

| Area | Technologies | Repository status |
|---|---|---|
| Algorithms | GRPO, PPO, DPO, RLHF | control-plane interfaces + GRPO math |
| Training | PyTorch, FSDP/FSDP2, Megatron-LM | integration configuration; GPU execution required |
| RL frameworks | VeRL | integration target for real GPU training |
| Rollout serving | vLLM, SGLang, TensorRT-LLM | backend registry/configuration; GPU execution required |
| Distributed runtime | Ray, Monarch, Miles | runtime registry + resource scheduler |
| Parallelism | data, tensor, pipeline, expert | topology configuration |
| MoE | expert placement / expert parallelism | architecture support; GPU benchmark required |
| Collectives | NCCL | Kubernetes/runtime configuration |
| Orchestration | Docker, Kubernetes | Dockerfile + GPU manifests |
| Reliability | checkpoint/recovery | atomic 10-step checkpoint manager |
