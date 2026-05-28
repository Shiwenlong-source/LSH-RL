# lsh-rl Example

This is the only example kept in the compact LSH-RL reproduction fork. It contains the two
main training pipelines from the paper's dialogue-to-action curriculum:

- `scripts/train_dialogue.sh`: dialogue-only RL training.
- `scripts/train_action.sh`: action-only RL training initialized from a dialogue adapter.

Both flows use four-card training by default and stratified scene sampling with a `2:1:1`
low/medium/high ratio. The code implements scene-grounded multi-role self-play, dialogue/action
typed transitions, long-short horizon reward fusion, and gated trajectory-to-turn credit
assignment. Ablation scripts, unrelated experiments, final results, logs, and checkpoints are
intentionally excluded. Reward logging and plotting code are kept.

## Run

Install the repository from the project root first:

```bash
cd LSH-RL
uv sync --extra gpu
```

Configure the local model and evaluator endpoint:

```bash
export AGL_BASE_MODEL=/path/to/base/model
export AGL_ADAPTER_PATH=/path/to/dialogue_or_dpo_adapter
export ROLEPLAY_ENV_BASE_URL=xxx
export ROLEPLAY_ENV_API_KEY=xxx
export ROLEPLAY_ENV_MODEL=xxx
```

Dialogue phase:

```bash
uv run bash examples/lsh-rl/scripts/train_dialogue.sh
```

Action phase:

```bash
export AGL_ACTION_INIT_ADAPTER_PATH=examples/lsh-rl/checkpoints/dialogue-RL/global_step_<N>/actor/lora_adapter
uv run bash examples/lsh-rl/scripts/train_action.sh
```

Checkpoints are written to:

```text
examples/lsh-rl/checkpoints/dialogue-RL/global_step_<N>/actor/lora_adapter
examples/lsh-rl/checkpoints/action-RL/global_step_<N>/actor/lora_adapter
```

Logs and reward curves are written to:

```text
examples/lsh-rl/logs/train_<timestamp>.log
examples/lsh-rl/logs/plots/
examples/lsh-rl/logs/images/
```

## Included Files

- `configs/base.env`: shared reward, logging, PPO, checkpoint, and VERL defaults.
- `configs/dialogue.env`: dialogue-only phase configuration.
- `configs/action.env`: action-only phase configuration, aligned with dialogue GPU and sampling settings.
- `scripts/train_dialogue.sh`: dialogue entrypoint.
- `scripts/train_action.sh`: action entrypoint.
- `run_train.sh`: shared training launcher.
- `train_stratified.py`: Agent-Lightning + VERL training entrypoint with difficulty-tier sampling.
- `train_roleplay_agent.py`: shared VERL config, checkpoint, resume, and task-loading utilities.
- `roleplay_agent.py`: PersonaArena-style multi-agent rollout and trace emission.
- `reward_evaluator.py`: short-term, long-term, and verifier reward implementation.
- `stratified_sampler.py`: low/medium/high batch sampler.
- `plot_training_metrics.py`: parses reward/training metrics from logs and writes CSV/PNG curves.
- `train_data/`: scene JSON files split into `low/`, `medium/`, and `high/`.
- `with_project_cache.sh`: local cache wrapper for Python/uv-launched commands.
