# Distributed LLM RL Post-Training & Rollout Platform

A portfolio-grade reference implementation for **distributed LLM post-training and rollout orchestration**. The project models the control plane around RL-style post-training: prompt batching, rollout generation, reward scoring, policy updates, experiment metrics, retries, and worker-level scheduling.

> This repository intentionally ships with a lightweight local simulator so it runs on a laptop without GPUs. The interfaces are designed so real backends such as PyTorch/Transformers, TRL, vLLM, Ray, or Kubernetes can be connected later.

## Why this project matters

It demonstrates AI/ML engineering skills that sit between model research and production systems:

- LLM post-training workflow design
- Parallel rollout generation
- Reward-model / rule-based scoring abstractions
- PPO/GRPO-style training-loop orchestration concepts
- Distributed worker scheduling and retries
- Metrics for reward, latency, throughput, and failures
- Reproducible configuration and testing
- Docker and GitHub Actions CI

## Architecture

```text
Prompts
  |
  v
Coordinator ---> Rollout Workers ---> Candidate Responses
  |                                   |
  |                                   v
  +------------------------------ Reward Scorer
                                      |
                                      v
                               Training Batch
                                      |
                                      v
                               Policy Updater
                                      |
                                      v
                                Metrics Store
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
python -m rl_platform.cli --prompts 32 --workers 4 --rollouts-per-prompt 2
```

## Example output

```text
prompts=32 rollouts=64 workers=4
mean_reward=0.73 throughput=118.4 rollouts/s failures=0
policy_update=simulated step=1
```

## Production extension points

The included interfaces can be replaced with:

- **Generation:** vLLM, SGLang, Hugging Face TGI, TensorRT-LLM
- **Training:** PyTorch FSDP, DeepSpeed, TRL
- **Distributed runtime:** Ray, Kubernetes Jobs
- **Tracking:** MLflow, Weights & Biases, Prometheus
- **Storage:** S3/GCS + Parquet

## Repository structure

```text
src/rl_platform/
  models.py        # core data contracts
  rollout.py       # rollout workers + coordinator
  reward.py        # reward scoring abstraction
  trainer.py       # post-training orchestration
  metrics.py       # run summary metrics
  cli.py           # runnable demo
configs/
  local.yaml
 tests/
```

## Roadmap

- [ ] Real Hugging Face causal LM adapter
- [ ] vLLM rollout backend
- [ ] Ray actor pool
- [ ] GRPO/PPO trainer adapter
- [ ] Distributed checkpointing
- [ ] Prometheus/Grafana dashboard
- [ ] Multi-node GPU benchmark report

## License

MIT
